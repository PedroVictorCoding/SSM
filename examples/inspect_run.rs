//! Read-only checkpoint health probe; writes no run files.
//! cargo run --release --example inspect_run -- runs/moe_hybrid
use burn::module::{ModuleVisitor, ParamId};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use sail::backend::{BACKEND_NAME, InferenceBackend as B, default_device};
use sail::train::load_run;

#[derive(Default)]
struct Health {
    total: usize,
    bad: usize,
    fingerprint: u64,
}
impl ModuleVisitor<B> for Health {
    fn visit_float<const D: usize>(&mut self, _id: ParamId, t: &Tensor<B, D>) {
        let v = t
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        for x in &v {
            self.fingerprint =
                self.fingerprint.wrapping_mul(1099511628211) ^ u64::from(x.to_bits());
        }
        self.total += v.len();
        self.bad += v.iter().filter(|x| !x.is_finite()).count();
    }
}
fn main() {
    let dir = std::env::args().nth(1).expect("run directory");
    let trace = std::env::args().any(|arg| arg == "--trace");
    let device = default_device();
    let (model, cfg, tok) = load_run::<B>(&dir, &device).expect("load checkpoint");
    let mut health = Health::default();
    model.visit(&mut health);
    println!(
        "backend={BACKEND_NAME} parameters={} non_finite={}",
        health.total, health.bad
    );
    println!("weights_fingerprint={:016x}", health.fingerprint);
    if health.bad > 0 {
        std::process::exit(2);
    }
    let encoded = tok.encode("ROMEO:\nThe king shall live.\n").unwrap();
    let len: usize = std::env::args()
        .nth(2)
        .map(|v| v.parse().expect("sequence length"))
        .unwrap_or(encoded.len() - 1);
    let batch: usize = std::env::args()
        .nth(3)
        .map(|v| v.parse().expect("batch size"))
        .unwrap_or(1);
    assert!(len > 0 && batch > 0);
    let encoded: Vec<_> = encoded.iter().copied().cycle().take(len + 1).collect();
    println!("batch={batch} sequence={len}");
    let x = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            encoded[..len]
                .iter()
                .cycle()
                .take(batch * len)
                .map(|x| *x as i64)
                .collect(),
            [batch, len],
        ),
        &device,
    );
    let target = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            encoded[1..]
                .iter()
                .cycle()
                .take(batch * len)
                .map(|x| *x as i64)
                .collect(),
            [batch * len],
        ),
        &device,
    );
    if trace {
        fn describe<const D: usize>(name: &str, tensor: Tensor<B, D>) {
            let data = tensor.into_data().convert::<f32>().to_vec::<f32>().unwrap();
            println!(
                "stage={name} first={:?} last={:?}",
                &data[..8.min(data.len())],
                &data[data.len().saturating_sub(8)..]
            );
        }
        let embedded = model.embedding.forward(x.clone());
        describe("embedding", embedded.clone());
        let normalized = model.blocks[0].norm.forward(embedded);
        describe("norm", normalized.clone());
        if let Some(mamba) = &model.blocks[0].mamba {
            describe("in_proj", mamba.in_proj.forward(normalized.clone()));
            {
                use burn::tensor::activation::{silu, softplus};
                let di = mamba.d_inner;
                let xz = mamba.in_proj.forward(normalized.clone());
                let xin = xz
                    .clone()
                    .narrow(2, 0, di)
                    .reshape([batch * len, di])
                    .reshape([batch, len, di]);
                let z = xz.narrow(2, di, di);
                let padded = Tensor::cat(
                    vec![Tensor::zeros([batch, mamba.d_conv - 1, di], &device), xin],
                    1,
                );
                let mut conv = mamba.conv_bias.val().unsqueeze::<3>();
                for j in 0..mamba.d_conv {
                    let w = mamba.conv_weight.val().narrow(1, j, 1).reshape([1, 1, di]);
                    conv = conv + padded.clone().narrow(1, j, len) * w;
                }
                describe("conv", conv.clone());
                let xc = silu(conv);
                describe("xc", xc.clone());
                let projected = mamba.x_proj.forward(xc.clone());
                describe("x_proj", projected.clone());
                let r = mamba.dt_rank;
                let n = mamba.d_state;
                let dt = projected.clone().narrow(2, 0, r);
                describe(
                    "delta_strided",
                    softplus(mamba.dt_proj.forward(dt.clone()), 1.0),
                );
                let delta = softplus(
                    mamba
                        .dt_proj
                        .forward(dt.reshape([batch * len, r]))
                        .reshape([batch, len, di]),
                    1.0,
                );
                describe("delta", delta.clone());
                let b = projected.clone().narrow(2, r, n);
                let c = projected.narrow(2, r + n, n);
                let a = mamba.a_log.val().exp().neg().unsqueeze::<4>();
                let da = (delta.clone().unsqueeze_dim::<4>(3) * a).exp();
                let dbx = (delta * xc.clone()).unsqueeze_dim::<4>(3) * b.unsqueeze_dim::<4>(2);
                describe("da", da.clone());
                describe("dbx", dbx.clone());
                let h = sail::model::scan::linear_scan(da, dbx, mamba.scan_chunk);
                describe("h", h.clone());
                let y = h.matmul(c.unsqueeze_dim::<4>(3)).reshape([batch, len, di])
                    + xc * mamba.d.val().unsqueeze::<3>();
                describe("y", y.clone());
                describe("z", z.clone());
                describe("gated", y.clone() * silu(z.clone()));
                describe("manual_mamba", mamba.out_proj.forward(y * silu(z)));
            }
            let mixed = mamba.forward(normalized);
            describe("mamba", mixed.clone());
            let residual = model.embedding.forward(x.clone()) + mixed;
            let mlp_input = model.blocks[0].mlp_norm.as_ref().unwrap().forward(residual);
            describe("mlp_norm", mlp_input.clone());
            if let Some(moe) = &model.blocks[0].moe {
                let routing = moe
                    .router
                    .forward(mlp_input.clone().reshape([batch * len, cfg.d_model]));
                let ids = routing
                    .clone()
                    .argmax(1)
                    .into_data()
                    .convert::<i64>()
                    .to_vec::<i64>()
                    .unwrap();
                println!(
                    "router_ids first={:?} last={:?}",
                    &ids[..8.min(ids.len())],
                    &ids[ids.len().saturating_sub(8)..]
                );
                describe("router", routing.reshape([batch, len, moe.num_experts]));
                describe("moe", moe.forward(mlp_input));
            }
        }
        let h = Tensor::<B, 4>::ones([1, 128, 128, 16], &device);
        let c = Tensor::<B, 4>::ones([1, 128, 16, 1], &device);
        let dot = h
            .clone()
            .matmul(c.clone())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let sum = (h * c.reshape([1, 128, 1, 16]))
            .sum_dim(3)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        println!(
            "ssm_contract_matmul_error={} sum_error={}",
            dot.iter().map(|v| (v - 16.0).abs()).fold(0.0_f32, f32::max),
            sum.iter().map(|v| (v - 16.0).abs()).fold(0.0_f32, f32::max)
        );
    }
    for (name, candidate) in [
        ("saved", model.clone()),
        ("chunked", model.clone().with_scan_chunk(64)),
        ("sequential", model.with_scan_chunk(0)),
    ] {
        for mode in ["prefill", "unobserved", "observed"] {
            let logits = match mode {
                "observed" => {
                    let (logits, _, probes) = candidate.forward_observed(x.clone(), true);
                    if trace {
                        for probe in &probes {
                            println!(
                                "probe={} first={:?}",
                                probe["name"],
                                &probe["values"].as_array().unwrap()[..4]
                            );
                        }
                    }
                    logits
                }
                "unobserved" => candidate.forward_with_aux(x.clone()).0,
                _ => candidate.forward(x.clone()),
            };
            let values = logits
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            let bad = values.iter().filter(|x| !x.is_finite()).count();
            let loss = CrossEntropyLossConfig::new()
                .init(&device)
                .forward(
                    logits.reshape([batch * len, cfg.vocab_size]),
                    target.clone(),
                )
                .into_scalar();
            println!("scan={name} mode={mode} logit_non_finite={bad} loss={loss}");
        }
    }
}
