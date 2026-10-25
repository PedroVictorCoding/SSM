use sail::data::bpe::BpeTokenizer;

#[test]
fn test_bpe_tokenizer_byte_fallback() {
    let tokenizer = BpeTokenizer::new_base();
    let text = "Hello world! 🌍 Testing UTF-8 bytes and symbols.";
    let encoded = tokenizer.encode(text);
    let decoded = tokenizer.decode(&encoded);
    assert_eq!(decoded, text);
    // In base mode, every byte is 1 token
    assert_eq!(encoded.len(), text.as_bytes().len());
}

#[test]
fn test_bpe_training_and_compression() {
    let corpus = "The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps again and again. \
                  Quick brown foxes are everywhere in the forest.";

    let base = BpeTokenizer::new_base();
    let base_tokens = base.encode(corpus);

    // Train BPE with 40 merges (258 base tokens + 40 merges = 298 vocab size)
    let bpe = BpeTokenizer::train(corpus, 298);
    let bpe_tokens = bpe.encode(corpus);
    let decoded = bpe.decode(&bpe_tokens);

    assert_eq!(
        decoded, corpus,
        "BPE round-trip decode must match original corpus"
    );
    assert!(
        bpe_tokens.len() < base_tokens.len(),
        "BPE must compress the sequence length (got {} tokens vs {} byte tokens)",
        bpe_tokens.len(),
        base_tokens.len()
    );

    // Verify subword reuse on new text
    let new_text = "The quick dog jumps";
    let new_encoded = bpe.encode(new_text);
    let new_decoded = bpe.decode(&new_encoded);
    assert_eq!(new_decoded, new_text);
}

#[test]
fn test_bpe_save_and_load() {
    let temp_path = "target/test_bpe_vocab.json";
    let corpus = "State space models and recurrent scans in pure Rust.";
    let bpe = BpeTokenizer::train(corpus, 280);

    bpe.save(temp_path).expect("Failed to save BPE tokenizer");
    let loaded = BpeTokenizer::load(temp_path).expect("Failed to load BPE tokenizer");

    let text = "Rust recurrent models.";
    assert_eq!(bpe.encode(text), loaded.encode(text));
    assert_eq!(loaded.decode(&loaded.encode(text)), text);

    let _ = std::fs::remove_file(temp_path);
}
