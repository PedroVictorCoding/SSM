use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::Distribution;
use sail::backend::TrainBackend;
use sail::model::mmap_moe::{MmapExpertStorage, StreamingMoe};
use sail::model::moe::MoeConfig;

#[test]
fn test_mmap_expert_streaming_equivalence() {
    let tmp_dir = std::env::temp_dir().join("ssm_test_mmap_moe");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let bin_path = tmp_dir.join("experts.bin");

    let device = Default::default();
    let d_model = 32;
    let d_hidden = 64;
    let num_experts = 4;
    let top_k = 2;

    let moe_cfg = MoeConfig::new(d_model, d_hidden)
        .with_num_experts(num_experts)
        .with_top_k(top_k)
        .with_dynamic_threshold(0.1);

    let dense_moe = moe_cfg.init::<TrainBackend>(&device);

    // Dump dense experts to memory-mapped storage
    let storage = MmapExpertStorage::dump_from_moe::<TrainBackend>(&bin_path, &dense_moe)
        .expect("Failed to dump experts to mmap");

    assert_eq!(storage.num_experts(), num_experts);
    assert_eq!(storage.d_model(), d_model);
    assert_eq!(storage.d_hidden(), d_hidden);

    // Instantiate StreamingMoe using the mmap storage and same router
    let streaming_moe = StreamingMoe::new(dense_moe.router.clone(), Arc::new(storage), top_k, 0.1);

    // Test input tensor
    let batch = 3;
    let seq_len = 8;
    let x = Tensor::<TrainBackend, 3>::random(
        [batch, seq_len, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Compute dense MoE outputs
    let (dense_out, dense_aux) = dense_moe.forward_with_aux(x.clone());

    // Compute streaming MoE outputs
    let (streaming_out, streaming_aux) = streaming_moe.forward_with_aux(x.clone());

    // Verify shapes
    assert_eq!(dense_out.shape().dims, [batch, seq_len, d_model]);
    assert_eq!(streaming_out.shape().dims, [batch, seq_len, d_model]);

    // Check numerical equivalence
    let diff = (dense_out - streaming_out)
        .abs()
        .max()
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        diff < 1e-5,
        "Max difference between dense MoE and Streaming MoE was {diff}, expected < 1e-5"
    );

    let aux_diff = (dense_aux - streaming_aux)
        .abs()
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        aux_diff < 1e-5,
        "Aux loss difference between dense and streaming was {aux_diff}, expected < 1e-5"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
