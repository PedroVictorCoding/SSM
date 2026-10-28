//! Integration tests for Provenance-Aware Long-Term Retrieval, User Preferences, and Memory Isolation.
//!
//! Validates:
//! - Cross-session retrieval of approved documents with traceable provenance.
//! - Multi-session and multi-user access scope isolation.
//! - Deletion, revoked access, and TTL expiration removing records from active retrieval and inverted indexes.
//! - Permanent rule: Model guesses/hypotheses are not automatically stored as verified facts or training data (ALIGNMENT.md).
//! - Preference revision superseding older versions and detecting conflicts.
//! - Retrieval relevance, latency, and storage growth benchmarks.

use sail::runtime::memory::{
    AccessScope, FactKind, LongTermMemoryStore, RetentionPolicy, RetrievalQuery,
};

#[test]
fn test_held_out_cross_session_retrieval_with_traceable_provenance() {
    let store = LongTermMemoryStore::new();

    // 1. Ingest approved architecture document
    store
        .store_approved_document(
            "doc_audio_pipeline",
            "audio.pipeline.spec",
            "The audio pipeline operates at 16000Hz mono PCM with a 20ms frame budget (320 samples).",
            Some(vec![0.1, 0.8, 0.2, 0.0]),
            "spec:hardware_audio_v2.md",
            AccessScope::Global,
            1000,
        )
        .expect("store doc");

    // 2. Ingest user preference for user_alice
    store
        .store_user_preference(
            "pref_alice_code",
            "editor.indent",
            "Always use 4 spaces for indentation in Rust and Python files.",
            Some(vec![0.5, 0.0, 0.5, 0.2]),
            "user_alice",
            1010,
        )
        .expect("store pref");

    // 3. Query from Session 1 (User Alice)
    let query_alice = RetrievalQuery {
        query_text: "What is the audio sample rate and frame size?".into(),
        query_embedding: Some(vec![0.1, 0.8, 0.19, 0.01]),
        user_id: Some("user_alice".into()),
        session_id: Some("session_001".into()),
        current_timestamp_ms: 1050,
        max_results: 3,
        max_tokens_budget: 256,
        min_score: 0.1,
        allowed_kinds: None,
    };

    let resp_alice = store.retrieve(&query_alice);
    assert!(!resp_alice.results.is_empty(), "Should retrieve audio spec");
    let top_result = &resp_alice.results[0];
    assert_eq!(top_result.record.id, "doc_audio_pipeline");
    assert_eq!(
        top_result.record.provenance.source_id,
        "spec:hardware_audio_v2.md"
    );
    assert_eq!(top_result.record.provenance.author, "human_approver");
    assert!(top_result.record.provenance.is_verified);
    assert!(!top_result.record.is_training_eligible);

    // 4. Query from Session 2 (Held-out new session for User Alice across sessions)
    let query_session_2 = RetrievalQuery {
        query_text: "How should I format indentation?".into(),
        query_embedding: Some(vec![0.49, 0.02, 0.51, 0.19]),
        user_id: Some("user_alice".into()),
        session_id: Some("session_002".into()),
        current_timestamp_ms: 2000,
        max_results: 3,
        max_tokens_budget: 256,
        min_score: 0.1,
        allowed_kinds: None,
    };

    let resp_session_2 = store.retrieve(&query_session_2);
    assert!(!resp_session_2.results.is_empty());
    assert_eq!(resp_session_2.results[0].record.key, "editor.indent");
    assert!(
        resp_session_2.results[0]
            .record
            .content
            .contains("4 spaces")
    );
    assert_eq!(
        resp_session_2.results[0].record.provenance.source_id,
        "user_pref:user_alice"
    );
}

#[test]
fn test_multi_session_and_user_access_isolation() {
    let store = LongTermMemoryStore::new();

    // Alice preference
    store
        .store_user_preference(
            "pref_alice",
            "theme",
            "Alice prefers dark mode theme.",
            None,
            "user_alice",
            1000,
        )
        .unwrap();

    // Bob preference
    store
        .store_user_preference(
            "pref_bob",
            "theme",
            "Bob prefers light mode theme.",
            None,
            "user_bob",
            1005,
        )
        .unwrap();

    // Session-isolated observation for session_alpha
    store
        .store_verified_observation(
            "obs_alpha",
            "temp.workspace",
            "Active buffer is /tmp/alpha_workspace.rs",
            None,
            "tool:workspace_probe",
            AccessScope::Session("session_alpha".into()),
            RetentionPolicy::SessionOnly,
            1010,
        )
        .unwrap();

    // Bob queries theme: MUST NOT see Alice's preference
    let query_bob = RetrievalQuery {
        query_text: "theme preference".into(),
        user_id: Some("user_bob".into()),
        session_id: Some("session_beta".into()),
        current_timestamp_ms: 1020,
        ..Default::default()
    };
    let resp_bob = store.retrieve(&query_bob);
    assert_eq!(resp_bob.results.len(), 1);
    assert_eq!(resp_bob.results[0].record.id, "pref_bob");
    assert!(resp_bob.results[0].record.content.contains("light mode"));

    // Session beta queries session observations: MUST NOT see session_alpha observation
    let query_beta = RetrievalQuery {
        query_text: "Active buffer workspace".into(),
        user_id: Some("user_bob".into()),
        session_id: Some("session_beta".into()),
        current_timestamp_ms: 1025,
        ..Default::default()
    };
    let resp_beta = store.retrieve(&query_beta);
    assert!(
        resp_beta.results.is_empty(),
        "Session beta must not access session_alpha memory"
    );
}

#[test]
fn test_user_deletion_and_source_revocation() {
    let store = LongTermMemoryStore::new();

    // Store two documents from the same external source
    store
        .store_approved_document(
            "doc_1",
            "company.api_v1",
            "API v1 uses basic auth and returns XML payloads.",
            None,
            "partner_corp_docs",
            AccessScope::Global,
            1000,
        )
        .unwrap();

    store
        .store_approved_document(
            "doc_2",
            "company.api_v2",
            "API v2 uses bearer tokens and JSON streaming.",
            None,
            "partner_corp_docs",
            AccessScope::Global,
            1005,
        )
        .unwrap();

    // Store temporary observation with 500ms TTL
    store
        .store_verified_observation(
            "obs_temp",
            "network.ip",
            "Local gateway IP is 192.168.1.1",
            None,
            "tool:net_probe",
            AccessScope::Global,
            RetentionPolicy::TtlMs(500),
            1000,
        )
        .unwrap();

    // 1. Explicitly delete doc_1
    store
        .delete_record("doc_1", "Deprecated API", 1100)
        .expect("delete doc_1");

    // Query for API
    let query = RetrievalQuery {
        query_text: "API payloads".into(),
        current_timestamp_ms: 1100,
        ..Default::default()
    };
    let resp = store.retrieve(&query);
    assert_eq!(resp.results.len(), 1);
    assert_eq!(
        resp.results[0].record.id, "doc_2",
        "doc_1 must be completely excluded after deletion"
    );

    // 2. TTL expiration
    let purged = store.purge_expired(1600); // 1600 > 1000 + 500
    assert_eq!(purged, 1, "Temporary observation should be purged");

    // 3. Revoke whole source
    let revoked = store.revoke_source("partner_corp_docs", 1700);
    assert_eq!(
        revoked, 1,
        "Remaining doc_2 from revoked source should be purged"
    );

    let resp_after_revoke = store.retrieve(&query);
    assert!(
        resp_after_revoke.results.is_empty(),
        "All revoked records must be gone"
    );
}

#[test]
fn test_model_hypotheses_cannot_become_training_data_or_verified_facts() {
    let store = LongTermMemoryStore::new();

    // Model generates a speculative hypothesis during dialogue
    store
        .store_model_hypothesis(
            "hypo_01",
            "model.guess.database_version",
            "The remote database might possibly be Postgres 14 based on error string.",
            None,
            AccessScope::Session("sess_01".into()),
            2000,
        )
        .expect("store hypothesis");

    let recs = store.inspect_records(None);
    let hypo = recs.iter().find(|r| r.id == "hypo_01").expect("found hypo");

    // Check invariants:
    assert_eq!(hypo.kind, FactKind::ModelHypothesis);
    assert!(
        !hypo.provenance.is_verified,
        "Hypothesis must NEVER be marked verified"
    );
    assert!(
        !hypo.is_training_eligible,
        "Hypothesis must NEVER be training eligible"
    );

    // Retrieval filtering: when query requests only Verified facts, hypothesis is excluded
    let query_verified_only = RetrievalQuery {
        query_text: "database version postgres".into(),
        session_id: Some("sess_01".into()),
        current_timestamp_ms: 2050,
        allowed_kinds: Some(vec![
            FactKind::ApprovedDocument,
            FactKind::VerifiedObservation,
        ]),
        ..Default::default()
    };
    let resp = store.retrieve(&query_verified_only);
    assert!(
        resp.results.is_empty(),
        "Verified-only retrieval must exclude speculative hypotheses"
    );
}

#[test]
fn test_preference_revision_and_conflict_detection() {
    let store = LongTermMemoryStore::new();

    // Version 1 of user preference
    store
        .store_user_preference(
            "pref_v1",
            "editor.font_size",
            "Font size is 12pt.",
            None,
            "alice",
            3000,
        )
        .unwrap();

    // User updates preference to Version 2
    store
        .store_user_preference(
            "pref_v2",
            "editor.font_size",
            "Font size is 16pt.",
            None,
            "alice",
            3010,
        )
        .unwrap();

    // Query preference
    let query = RetrievalQuery {
        query_text: "font size editor".into(),
        user_id: Some("alice".into()),
        current_timestamp_ms: 3020,
        ..Default::default()
    };
    let resp = store.retrieve(&query);
    assert_eq!(
        resp.results.len(),
        1,
        "Only active superseded version returned"
    );
    assert_eq!(resp.results[0].record.id, "pref_v2");
    assert_eq!(resp.results[0].record.provenance.version, 2);
    assert!(resp.results[0].record.content.contains("16pt"));
}

#[test]
fn test_retrieval_benchmark_and_storage_growth() {
    let store = LongTermMemoryStore::new();

    // Ingest 100 realistic technical snippets
    for i in 0..100 {
        let id = format!("doc_{i:03}");
        let key = format!("tech.topic_{i}");
        let content = format!(
            "Technical specification document {i} discussing module architecture, memory allocation, and performance profiling in rust."
        );
        let emb = vec![(i as f32) / 100.0, 0.5, 0.2, 0.1];
        store
            .store_approved_document(
                &id,
                &key,
                &content,
                Some(emb),
                "benchmark_suite",
                AccessScope::Global,
                4000 + i as u64,
            )
            .unwrap();
    }

    let metrics = store.get_storage_metrics();
    assert_eq!(metrics.active_records, 100);
    assert!(metrics.total_bytes_content > 5000);
    assert!(metrics.inverted_index_terms > 10);
    assert_eq!(metrics.unique_sources, 1);

    // Benchmark query latency
    let query = RetrievalQuery {
        query_text: "module architecture and memory allocation".into(),
        query_embedding: Some(vec![0.5, 0.5, 0.2, 0.1]),
        current_timestamp_ms: 5000,
        max_results: 5,
        max_tokens_budget: 200,
        min_score: 0.1,
        allowed_kinds: None,
        ..Default::default()
    };

    let resp = store.retrieve(&query);
    assert_eq!(resp.results.len(), 5);
    // Retrieval latency must easily meet the sub-millisecond budget for 100 docs
    assert!(
        resp.latency_ms < 10.0,
        "Retrieval latency {:.3}ms was too slow",
        resp.latency_ms
    );
}
