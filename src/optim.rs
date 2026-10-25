//! Preserve Burn 0.16's per-parameter norm clipping without host scalar reads.
use crate::model::HybridLm;
use burn::{
    module::{Module, ModuleVisitor, ParamId},
    optim::GradientsParams,
    tensor::{Tensor, backend::AutodiffBackend},
};

pub fn clip_gradients_on_device<B: AutodiffBackend>(
    model: &HybridLm<B>,
    mut grads: GradientsParams,
    threshold: f32,
) -> GradientsParams {
    assert!(threshold.is_finite() && threshold >= 0.0);
    struct Clip<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        threshold: f32,
        marker: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Clip<'_, B> {
        fn visit_float<const D: usize>(&mut self, id: ParamId, _: &Tensor<B, D>) {
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
                self.grads.register(id, clip_tensor(grad, self.threshold));
            }
        }
    }
    model.visit(&mut Clip::<B> {
        grads: &mut grads,
        threshold,
        marker: std::marker::PhantomData,
    });
    grads
}

fn clip_tensor<B: burn::tensor::backend::Backend, const D: usize>(
    grad: Tensor<B, D>,
    threshold: f32,
) -> Tensor<B, D> {
    let norm = grad.clone().powf_scalar(2.0).sum().sqrt();
    // A zero threshold zeros nonzero gradients and leaves zero gradients zero.
    // The positive floor also avoids 0/0 for a zero norm.
    let denominator = norm.clamp_min(threshold.max(f32::MIN_POSITIVE));
    let scale = denominator.recip().mul_scalar(threshold);
    grad * scale.reshape([1; D])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{InferenceBackend as B, default_device};
    use burn::{grad_clipping::GradientClippingConfig, tensor::TensorData};
    #[test]
    fn device_clipping_matches_burn_for_zero_small_and_large_norms() {
        let device = default_device();
        for values in [
            [0.0, 0.0, 0.0, 0.0],
            [0.01, -0.02, 0.03, -0.04],
            [3.0, -4.0, 6.0, -8.0],
        ] {
            for threshold in [0.0, 0.1, 1.0, 100.0] {
                let tensor =
                    Tensor::<B, 2>::from_data(TensorData::new(values.to_vec(), [2, 2]), &device);
                let expected = GradientClippingConfig::Norm(threshold)
                    .init()
                    .clip_gradient(tensor.clone())
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let actual = clip_tensor(tensor, threshold)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                for (a, b) in actual.iter().zip(expected) {
                    assert!(
                        a.is_finite() && (a - b).abs() <= 2e-6 + b.abs() * 2e-6,
                        "actual={a}, expected={b}"
                    );
                }
            }
        }
    }
}
