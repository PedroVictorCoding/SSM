//! Storage Lifecycle, Bounded Experience Replay, and Ingestion Receipt Verification (DATA-01)

use std::fs;
use std::path::PathBuf;

use sail::data::{CharTokenizer, IncomingDirectoryWatcher, StreamConfig, StreamDataset, Tokenizer};

#[test]
fn test_bounded_experience_replay_buffer_capacity() {
    let config = StreamConfig {
        replay_ratio: 0.3,
        min_tokens_to_train: 10,
        scan_interval_ms: 100,
        max_replay_tokens: 100, // strict RAM budget
    };

    let dataset = StreamDataset::new(Vec::new(), config);

    // Initial batch: 60 tokens
    let tokens1: Vec<u32> = (0..60).collect();
    dataset.append_tokens(&tokens1);
    assert_eq!(dataset.total_tokens(), 60);
    let (hist, new_tokens) = dataset.token_distribution();
    assert_eq!(hist, 0);
    assert_eq!(new_tokens, 60);

    // Second batch: 60 tokens -> total attempted = 120 > 100 cap
    let tokens2: Vec<u32> = (60..120).collect();
    dataset.append_tokens(&tokens2);

    // Assert strictly bounded to max_replay_tokens
    assert_eq!(dataset.total_tokens(), 100);

    // The oldest 20 tokens (0..20) must have been drained from the front
    let snapshot = dataset.snapshot();
    assert_eq!(snapshot.tokens.len(), 100);
    assert_eq!(snapshot.tokens[0], 20); // first retained token
    assert_eq!(snapshot.tokens[99], 119); // last appended token

    // Watermark was 60, after 20 pruned it should be 40
    assert_eq!(snapshot.new_data_watermark, 40);
    let (hist2, new_tokens2) = dataset.token_distribution();
    assert_eq!(hist2, 40);
    assert_eq!(new_tokens2, 60);
}

#[test]
fn test_receipt_based_cleanup_protocol_and_spool_binding() {
    let temp_dir = std::env::temp_dir().join(format!("sail-lifecycle-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let incoming_dir = temp_dir.join("incoming");
    fs::create_dir_all(&incoming_dir).unwrap();

    // 1. Setup spool metadata with a unique spool UUID
    let spool_dir = incoming_dir.join(".hf_streamer");
    fs::create_dir_all(&spool_dir).unwrap();
    let spool_uuid = "spool_test_uuid_12345";
    let state_json = serde_json::json!({
        "format": 1,
        "id": spool_uuid,
        "datasets_version": "2.0.0",
        "filters": {"min_chars": 5, "max_record_bytes": 1048576},
        "sources": {},
        "pending": null
    });
    fs::write(
        spool_dir.join("state.json"),
        serde_json::to_vec(&state_json).unwrap(),
    )
    .unwrap();

    // 2. Create staged chunk files
    let chunk1 = incoming_dir.join(format!("hf_{spool_uuid}_src0_0000000001.txt"));
    let chunk2 = incoming_dir.join(format!("hf_{spool_uuid}_src0_0000000002.txt"));
    fs::write(&chunk1, "hello stream world alpha").unwrap();
    fs::write(&chunk2, "hello stream world beta").unwrap();

    let mut watcher = IncomingDirectoryWatcher::new(&incoming_dir).unwrap();
    let tokenizer = Tokenizer::Char(CharTokenizer::from_chars(vec![
        'h', 'e', 'l', 'o', ' ', 's', 't', 'r', 'a', 'm', 'w', 'd', 'p', 'b',
    ]));
    let dataset = StreamDataset::new(Vec::new(), StreamConfig::default());

    // 3. Ingest files
    let count = watcher.scan_and_ingest(&dataset, &tokenizer);
    assert!(count > 0);
    assert_eq!(watcher.processed_files().len(), 2);

    // 4. Verify receipt structure
    let receipt = watcher.ingestion_receipt();
    assert_eq!(
        receipt.get("spool_id").unwrap().as_str().unwrap(),
        spool_uuid
    );
    let processed = receipt.get("processed_files").unwrap().as_array().unwrap();
    assert_eq!(processed.len(), 2);
    assert!(
        processed
            .iter()
            .any(|v| v.as_str().unwrap().contains("0000000001.txt"))
    );
    assert!(
        processed
            .iter()
            .any(|v| v.as_str().unwrap().contains("0000000002.txt"))
    );

    // 5. Simulate checkpoint restoration of processed files
    let mut restored_watcher = IncomingDirectoryWatcher::new(&incoming_dir).unwrap();
    let processed_paths: Vec<PathBuf> = watcher.processed_files();
    restored_watcher
        .restore_processed_files(&processed_paths)
        .unwrap();
    assert_eq!(restored_watcher.processed_files().len(), 2);

    // New scan should not re-ingest already acknowledged files
    let duplicate_count = restored_watcher.scan_and_ingest(&dataset, &tokenizer);
    assert_eq!(duplicate_count, 0);

    let _ = fs::remove_dir_all(&temp_dir);
}
