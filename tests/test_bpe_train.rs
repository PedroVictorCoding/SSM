use sail::backend::TrainBackend;
use sail::data::Tokenizer;
use sail::model::HybridLmConfig;
use sail::train::{TrainConfig, load_run, train};

#[test]
fn test_bpe_training_and_generation() {
    let tmp_dir = std::env::temp_dir().join("ssm_test_bpe_train");
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let text = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles";
    let device = Default::default();

    let model_cfg = HybridLmConfig::new(300)
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8)
        .with_expand(2);

    let train_cfg = TrainConfig::new(model_cfg)
        .with_steps(5)
        .with_batch_size(2)
        .with_seq_len(16)
        .with_use_bpe(true)
        .with_bpe_vocab_size(280)
        .with_eval_every(5)
        .with_log_every(5);

    let summary =
        train::<TrainBackend>(train_cfg, text, &tmp_dir, &device).expect("training failed");
    assert_eq!(summary.steps_run, 5);

    // Verify saved tokenizer
    let (model, loaded_cfg, tokenizer) =
        load_run::<TrainBackend>(&tmp_dir, &device).expect("load_run failed");
    match tokenizer {
        Tokenizer::Bpe(bpe) => {
            assert!(bpe.vocab_size() >= 258);
            let encoded = bpe.encode("noble mind");
            let decoded = bpe.decode(&encoded);
            assert_eq!(decoded, "noble mind");
        }
        Tokenizer::Char(_) => panic!("expected Bpe tokenizer"),
    }
    assert_eq!(loaded_cfg.d_model, 32);
    assert_eq!(model.total_params(), model.active_params_per_token());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
