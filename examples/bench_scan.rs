//! Portable scan experiment; reports median forward+backward latency.
//! cargo run --release --example bench_scan -- 2 64 8 7
use burn::prelude::*;
use burn::tensor::Distribution;
use sail::backend::{BACKEND_NAME, TrainBackend as B, default_device};
use sail::model::scan::{chunked_parallel_scan, work_efficient_scan};
use std::time::Instant;

fn run(a: &Tensor<B, 4>, b: &Tensor<B, 4>, tree: bool) -> f64 {
    let start = Instant::now();
    let h = if tree {
        work_efficient_scan(a.clone(), b.clone())
    } else {
        chunked_parallel_scan(a.clone(), b.clone(), 64)
    };
    let gradients = h.sum().backward();
    let _ = a.grad(&gradients).unwrap().sum().into_scalar();
    let _ = b.grad(&gradients).unwrap().sum().into_scalar();
    start.elapsed().as_secs_f64() * 1000.0
}
fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|v| v.parse().expect("positive integer"))
        .collect();
    let batch = args.first().copied().unwrap_or(2);
    let width = args.get(1).copied().unwrap_or(64);
    let state = args.get(2).copied().unwrap_or(8);
    let repeats = args.get(3).copied().unwrap_or(7);
    assert!(batch > 0 && width > 0 && state > 0 && repeats > 0);
    let device = default_device();
    B::seed(42);
    println!(
        "Backend: {BACKEND_NAME}; B={batch}, D={width}, N={state}; chunk=64; {repeats} repeats; median forward+backward"
    );
    println!("L,chunked_ms,tree_ms,speedup");
    for len in [16, 32, 64, 127, 128, 256] {
        let a = Tensor::<B, 4>::random(
            [batch, len, width, state],
            Distribution::Uniform(0.5, 1.0),
            &device,
        )
        .require_grad();
        let b = Tensor::<B, 4>::random(
            [batch, len, width, state],
            Distribution::Normal(0.0, 1.0),
            &device,
        )
        .require_grad();
        run(&a, &b, false);
        run(&a, &b, true);
        let mut old = Vec::new();
        let mut new = Vec::new();
        for i in 0..repeats {
            if i % 2 == 0 {
                old.push(run(&a, &b, false));
                new.push(run(&a, &b, true));
            } else {
                new.push(run(&a, &b, true));
                old.push(run(&a, &b, false));
            }
        }
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        let baseline = old[repeats / 2];
        let tree = new[repeats / 2];
        println!("{len},{baseline:.3},{tree:.3},{:.3}", baseline / tree);
    }
}
