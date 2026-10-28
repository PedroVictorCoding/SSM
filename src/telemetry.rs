//! Real parameter snapshots, taken immediately before the optimizer update.
use crate::model::{HybridLm, mlp::SwiGlu};
use burn::{
    module::Param, nn::Linear, optim::GradientsParams, prelude::*, tensor::backend::AutodiffBackend,
};
use serde_json::{Value, json};
use std::{fs, io, path::Path};

fn stats(values: &[f32]) -> Value {
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut non_finite = 0usize;

    for &v in values {
        if v.is_finite() {
            let vf = v as f64;
            count += 1;
            sum += vf;
            sum_sq += vf * vf;
            if vf < min {
                min = vf;
            }
            if vf > max {
                max = vf;
            }
        } else {
            non_finite += 1;
        }
    }

    let n = count.max(1) as f64;
    let min_val = if count > 0 { Some(min) } else { None };
    let max_val = if count > 0 { Some(max) } else { None };

    json!({
        "mean": sum / n,
        "rms": (sum_sq / n).sqrt(),
        "min": min_val,
        "max": max_val,
        "non_finite": non_finite,
    })
}

struct Capture<'a> {
    grads: Option<&'a GradientsParams>,
    max_values: usize,
    tensors: Vec<Value>,
}
impl Capture<'_> {
    fn param<B: AutodiffBackend, const D: usize>(
        &mut self,
        name: &str,
        param: &Param<Tensor<B, D>>,
    ) {
        let tensor = param.val();
        let shape = tensor.dims();
        let values = tensor
            .inner()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("f32 telemetry");
        let gradient = self
            .grads
            .and_then(|g| g.get::<B::InnerBackend, D>(param.id))
            .map(|g| {
                stats(
                    &g.into_data()
                        .convert::<f32>()
                        .to_vec::<f32>()
                        .expect("f32 gradients"),
                )
            });
        let stride = if self.max_values == 0 {
            1
        } else {
            values.len().div_ceil(self.max_values).max(1)
        };
        let visible: Vec<Option<f32>> = values
            .iter()
            .step_by(stride)
            .map(|v| v.is_finite().then_some(*v))
            .collect();
        self.tensors.push(
            json!({"name": name, "shape": shape.to_vec(), "count": values.len(),
            "stats": stats(&values), "gradient": gradient, "stride": stride, "values": visible}),
        );
    }
    fn linear<B: AutodiffBackend>(&mut self, name: &str, linear: &Linear<B>) {
        self.param(&format!("{name}.weight"), &linear.weight);
        if let Some(bias) = &linear.bias {
            self.param(&format!("{name}.bias"), bias);
        }
    }
    fn mlp<B: AutodiffBackend>(&mut self, name: &str, mlp: &SwiGlu<B>) {
        self.linear(&format!("{name}.gate"), &mlp.gate);
        self.linear(&format!("{name}.up"), &mlp.up);
        self.linear(&format!("{name}.down"), &mlp.down);
    }
}

pub fn publish<B: AutodiffBackend>(
    model: &HybridLm<B>,
    grads: Option<&GradientsParams>,
    dir: &Path,
    step: usize,
    max_values: usize,
    activations: Vec<Value>,
) -> io::Result<()> {
    let mut c = Capture {
        grads,
        max_values,
        tensors: Vec::new(),
    };
    c.param("embedding.weight", &model.embedding.weight);
    for (i, block) in model.blocks.iter().enumerate() {
        let name = format!("blocks.{i}");
        c.param(&format!("{name}.norm.gamma"), &block.norm.gamma);
        if let Some(m) = &block.mamba {
            let n = format!("{name}.mamba");
            c.linear(&format!("{n}.in_proj"), &m.in_proj);
            c.param(&format!("{n}.conv_weight"), &m.conv_weight);
            c.param(&format!("{n}.conv_bias"), &m.conv_bias);
            c.linear(&format!("{n}.x_proj"), &m.x_proj);
            c.linear(&format!("{n}.dt_proj"), &m.dt_proj);
            c.param(&format!("{n}.a_log"), &m.a_log);
            c.param(&format!("{n}.d"), &m.d);
            c.linear(&format!("{n}.out_proj"), &m.out_proj);
        }
        if let Some(a) = &block.attn {
            c.linear(&format!("{name}.attn.qkv"), &a.qkv);
            c.linear(&format!("{name}.attn.out"), &a.out);
        }
        if let Some(norm) = &block.mlp_norm {
            c.param(&format!("{name}.mlp_norm.gamma"), &norm.gamma);
        }
        if let Some(mlp) = &block.mlp {
            c.mlp(&format!("{name}.mlp"), mlp);
        }
        if let Some(moe) = &block.moe {
            c.linear(&format!("{name}.moe.router"), &moe.router);
            for (j, expert) in moe.experts.iter().enumerate() {
                c.mlp(&format!("{name}.moe.experts.{j}"), expert);
            }
        }
    }
    c.param("norm_f.gamma", &model.norm_f.gamma);
    c.linear("lm_head", &model.lm_head);
    let snapshot = json!({"step": step, "weights_step": step.saturating_sub(1),
        "timing": "before_optimizer", "max_values": max_values, "tensors": c.tensors, "activations": activations});
    let tmp_path = dir.join("neural.json.tmp");
    let file = fs::File::create(&tmp_path)?;
    let mut writer = io::BufWriter::with_capacity(64 * 1024, file);
    serde_json::to_writer(&mut writer, &snapshot)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    use std::io::Write;
    writer.flush()?;
    fs::rename(tmp_path, dir.join("neural.json"))
}
