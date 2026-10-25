use sail::backend::TrainBackend;
use sail::data::{BpeTokenizer, Tokenizer};
use sail::generate::{SamplingConfig, compute_normalized_entropy, generate_stateful_with_tools};
use sail::model::HybridLmConfig;
use sail::tool::{ToolRegistry, WebSearchTool};

#[test]
fn test_compute_normalized_entropy_extremes() {
    // Sharp distribution (100% confidence on index 0): entropy must be 0.0
    let sharp_logits = vec![100.0f32, -50.0, -50.0, -50.0];
    let sharp_ent = compute_normalized_entropy(&sharp_logits);
    assert!(sharp_ent < 1e-4, "Expected near 0 entropy, got {sharp_ent}");

    // Uniform distribution (maximum uncertainty): entropy must be ~1.0
    let flat_logits = vec![5.0f32; 16];
    let flat_ent = compute_normalized_entropy(&flat_logits);
    assert!(
        (flat_ent - 1.0).abs() < 1e-4,
        "Expected 1.0 entropy, got {flat_ent}"
    );
}

#[test]
fn test_tool_registry_web_search() {
    let registry = ToolRegistry::new();
    let res = registry.dispatch("web_search(\"capital of france\")");
    assert!(res.contains("Paris"));

    let res_league = registry.dispatch("web_search: premier league");
    assert!(res_league.contains("Arsenal") || res_league.contains("Premier League"));
}

#[test]
fn test_stateful_generation_with_tool_calling_and_entropy() {
    let device = Default::default();
    let tokenizer = Tokenizer::Bpe(BpeTokenizer::new_base());

    let model_cfg = HybridLmConfig::new(tokenizer.vocab_size())
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8)
        .with_expand(2);

    let model = model_cfg.init::<TrainBackend>(&device);
    let registry = ToolRegistry::new();

    // Prompt containing an explicit tool call to search for the speed of light
    let prompt_text = "Fact check: <tool_call>web_search(\"speed of light\")</tool_call>";
    let prompt_ids = tokenizer.encode(prompt_text).expect("encode prompt");

    let cfg = SamplingConfig {
        temperature: 0.0, // greedy
        top_k: 0,
        seed: 42,
        entropy_threshold: Some(0.99), // Only trigger on extreme uncertainty
        max_uncertain_tokens: 3,
    };

    let (out_tokens, stats) = generate_stateful_with_tools(
        &model,
        &tokenizer,
        &registry,
        &prompt_ids,
        12,
        &cfg,
        &device,
    );

    let decoded = tokenizer.decode(&out_tokens);
    assert!(stats.tool_calls_executed >= 1, "Tool call was not executed");
    assert!(
        decoded.contains("<tool_response>"),
        "Output missing tool response"
    );
    assert!(
        decoded.contains("299,792,458"),
        "Output missing verified tool search data"
    );
    assert_eq!(stats.token_entropies.len(), stats.new_tokens);
    assert!(stats.average_entropy() >= 0.0 && stats.average_entropy() <= 1.0);
}

#[test]
fn test_entropy_triggered_anti_hallucination_verification() {
    let device = Default::default();
    let tokenizer = Tokenizer::Bpe(BpeTokenizer::new_base());

    let model_cfg = HybridLmConfig::new(tokenizer.vocab_size())
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8);

    let model = model_cfg.init::<TrainBackend>(&device);
    let mut registry = ToolRegistry::new();
    registry.register(std::sync::Arc::new(WebSearchTool::new().with_fact(
        "quantum",
        "Quantum computers leverage superposition and entanglement.",
    )));

    let prompt_ids = tokenizer.encode("The nature of quantum").expect("encode");

    // Force entropy threshold to 0.05 so any normal prediction triggers verification
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_k: 0,
        seed: 42,
        entropy_threshold: Some(0.05),
        max_uncertain_tokens: 1,
    };

    let (out_tokens, stats) =
        generate_stateful_with_tools(&model, &tokenizer, &registry, &prompt_ids, 8, &cfg, &device);

    let decoded = tokenizer.decode(&out_tokens);
    assert!(
        stats.tool_calls_executed >= 1,
        "Entropy threshold should trigger automatic verification"
    );
    assert!(
        decoded.contains("<tool_response>"),
        "Decoded output should contain verified search injection"
    );
}
