use std::fs;
use std::path::PathBuf;

use rand::SeedableRng;
use rand::rngs::StdRng;
use sail::backend::{TrainBackend, default_device};
use sail::data::{CharTokenizer, IncomingDirectoryWatcher, StreamConfig, StreamDataset, Tokenizer};

#[test]
fn test_stream_dataset_append_and_distribution() {
    let initial_tokens = vec![1, 2, 3, 4, 5];
    let config = StreamConfig {
        replay_ratio: 0.4,
        min_tokens_to_train: 10,
        scan_interval_ms: 100,
    };
    let dataset = StreamDataset::new(initial_tokens, config);

    assert_eq!(dataset.total_tokens(), 5);
    let (hist, fresh) = dataset.token_distribution();
    assert_eq!(hist, 5);
    assert_eq!(fresh, 0);

    // Append new tokens
    dataset.append_tokens(&[6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(dataset.total_tokens(), 12);
    let (hist2, fresh2) = dataset.token_distribution();
    assert_eq!(hist2, 5);
    assert_eq!(fresh2, 7);
}

#[test]
fn test_stream_dataset_experience_replay_batch_sampling() {
    let initial_tokens: Vec<u32> = (0..100).collect();
    let config = StreamConfig {
        replay_ratio: 0.5,
        min_tokens_to_train: 20,
        scan_interval_ms: 100,
    };
    let dataset = StreamDataset::new(initial_tokens, config);
    // Append 100 new tokens
    let new_tokens: Vec<u32> = (100..200).collect();
    dataset.append_tokens(&new_tokens);

    assert_eq!(dataset.total_tokens(), 200);

    let mut rng = StdRng::seed_from_u64(42);
    let device = default_device();
    let batch_size = 8;
    let seq_len = 16;

    let batch = dataset
        .sample_batch::<TrainBackend, _>(batch_size, seq_len, &mut rng, &device)
        .expect("batch sampling should succeed");

    assert_eq!(batch.inputs.dims(), [batch_size, seq_len]);
    assert_eq!(batch.targets.dims(), [batch_size, seq_len]);
}

#[test]
fn test_incoming_directory_watcher_ingestion() {
    let test_dir = PathBuf::from("target/test_incoming_watcher");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    let mut watcher = IncomingDirectoryWatcher::new(&test_dir).unwrap();

    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n",
    ));
    let dataset = StreamDataset::new(Vec::new(), StreamConfig::default());

    assert_eq!(dataset.total_tokens(), 0);

    // Drop first file
    let file1 = test_dir.join("drop1.txt");
    fs::write(&file1, "hello stream world\n").unwrap();

    let added1 = watcher.scan_and_ingest(&dataset, &tokenizer);
    assert!(added1 > 0);
    assert_eq!(dataset.total_tokens(), added1);

    // Scan again without any new files; should add 0
    let added_none = watcher.scan_and_ingest(&dataset, &tokenizer);
    assert_eq!(added_none, 0);

    // Drop second file
    let file2 = test_dir.join("drop2.jsonl");
    fs::write(&file2, "continual learning active\n").unwrap();

    let added2 = watcher.scan_and_ingest(&dataset, &tokenizer);
    assert!(added2 > 0);
    assert_eq!(dataset.total_tokens(), added1 + added2);

    let _ = fs::remove_dir_all(&test_dir);
}
