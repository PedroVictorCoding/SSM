//! Compile-time backend selection.
//!
//! Choose with cargo features (priority `cuda` > `wgpu` > `ndarray`):
//! - `cargo run --release -- ...`                                   (ndarray, CPU, default)
//! - `cargo run --release --no-default-features --features wgpu -- ...`  (Vulkan/Metal/DX12)
//! - `cargo run --release --no-default-features --features cuda -- ...`  (NVIDIA CUDA)

use burn::backend::Autodiff;

#[cfg(feature = "cuda")]
mod selected {
    pub type Inner = burn::backend::CudaJit<f32, i32>;
    pub const NAME: &str = "cuda";
    pub fn device() -> burn::backend::cuda_jit::CudaDevice {
        burn::backend::cuda_jit::CudaDevice::default()
    }
}

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
mod selected {
    #[cfg(not(feature = "wgpu-unfused"))]
    pub type Inner = burn::backend::Wgpu<f32, i32>;
    #[cfg(feature = "wgpu-unfused")]
    pub type Inner = burn::backend::wgpu::JitBackend<
        burn::backend::wgpu::WgpuRuntime<burn::backend::wgpu::Wgsl>,
        f32,
        i32,
        u32,
    >;
    #[cfg(not(feature = "wgpu-unfused"))]
    pub const NAME: &str = "wgpu";
    #[cfg(feature = "wgpu-unfused")]
    pub const NAME: &str = "wgpu (unfused)";
    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        static DEVICE: std::sync::OnceLock<burn::backend::wgpu::WgpuDevice> =
            std::sync::OnceLock::new();
        DEVICE
            .get_or_init(|| {
                use burn::backend::wgpu::{AutoGraphicsApi, WgpuDevice, init_setup};
                let device = WgpuDevice::default();
                let setup = init_setup::<AutoGraphicsApi>(&device, Default::default());
                let info = setup.adapter.get_info();
                println!(
                    "GPU adapter: {} | {:?} | {:?} | driver: {} {}",
                    info.name, info.device_type, info.backend, info.driver, info.driver_info
                );
                device
            })
            .clone()
    }
}

#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "cuda")))]
mod selected {
    pub type Inner = burn::backend::NdArray<f32>;
    pub const NAME: &str = "ndarray (CPU)";
    pub fn device() -> burn::backend::ndarray::NdArrayDevice {
        burn::backend::ndarray::NdArrayDevice::Cpu
    }
}

#[cfg(not(any(feature = "ndarray", feature = "wgpu", feature = "cuda")))]
compile_error!("Enable one backend feature: `ndarray`, `wgpu` or `cuda`.");

/// Backend used for inference / evaluation.
pub type InferenceBackend = selected::Inner;
/// Backend used for training (autodiff wrapper around [`InferenceBackend`]).
pub type TrainBackend = Autodiff<InferenceBackend>;
/// Human readable backend name.
pub const BACKEND_NAME: &str = selected::NAME;

/// Default device for the selected backend.
pub fn default_device() -> <InferenceBackend as burn::tensor::backend::Backend>::Device {
    selected::device()
}
