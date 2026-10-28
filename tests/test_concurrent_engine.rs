use std::fs;
use std::path::PathBuf;

use sail::backend::{InferenceBackend, default_device};
use sail::data::{CharTokenizer, Tokenizer};
use sail::engine::Engine;
use sail::export::pack_to_artifact;
use sail::generate::SamplingConfig;
use sail::model::HybridLmConfig;

#[test]
fn test_concurrent_parallel_engine_and_hot_reload() {
    let test_dir = PathBuf::from("target/test_concurrent_engine");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    let model_path = test_dir.join("model.sea");
    let device = default_device();

    let text = "ABC abc 123 !? \n";
    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(text));
    let cfg = HybridLmConfig::new(tokenizer.vocab_size())
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8)
        .with_d_conv(2);

    let model = cfg.init::<InferenceBackend>(&device);

    // Export generation 1
    pack_to_artifact(&model, &cfg, &tokenizer, &model_path).unwrap();

    // 1. Initialize Concurrent Engine
    let engine = Engine::<InferenceBackend>::open(&model_path, &device)
        .expect("Engine must open valid .sea artifact");

    assert_eq!(engine.active_generation().generation_id, 1);

    // 2. Single thread inference
    let sampling = SamplingConfig {
        temperature: 0.8,
        top_k: 2,
        seed: 42,
        ..Default::default()
    };
    let (reply, stats) = engine
        .generate("A", 5, &sampling)
        .expect("Generation should succeed");
    assert!(reply.starts_with('A'));
    assert_eq!(stats.new_tokens, 5);

    // 3. Parallel inference across multiple OS threads concurrently
    let prompts = vec![
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "1".to_string(),
    ];
    let parallel_results = engine.generate_parallel(&prompts, 5, &sampling);
    assert_eq!(parallel_results.len(), 4);
    for (i, res) in parallel_results.iter().enumerate() {
        let (txt, st) = res.as_ref().expect("parallel worker should succeed");
        assert!(txt.starts_with(&prompts[i]));
        assert_eq!(st.new_tokens, 5);
    }

    // 4. Simulate background training saving an updated model.sea (Generation 2)
    std::thread::sleep(std::time::Duration::from_millis(50));
    let updated_model = cfg.init::<InferenceBackend>(&device);
    pack_to_artifact(&updated_model, &cfg, &tokenizer, &model_path).unwrap();

    // Check and hot-reload non-blockingly
    let reloaded = engine
        .check_and_reload()
        .expect("Check reload should succeed");
    assert!(reloaded, "Engine must detect the new model on disk");
    assert_eq!(engine.active_generation().generation_id, 2);

    // Run parallel inference on newly reloaded generation
    let post_reload_results = engine.generate_parallel(&prompts, 5, &sampling);
    assert_eq!(post_reload_results.len(), 4);
    for res in post_reload_results {
        assert!(res.is_ok());
    }

    let _ = fs::remove_dir_all(&test_dir);
}
