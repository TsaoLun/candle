use candle_core::{Device, Tensor};
use std::time::Instant;

fn benchmark_matmul(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    name: &str,
    num_samples: usize,
) -> (f64, f64) {
    let device = Device::Cpu;
    let shape_lhs = (batch, m, k);
    let shape_rhs = (batch, k, n);
    
    // === PREPARE: 创建输入 (只做一次) ===
    let lhs = Tensor::randn(0.0f32, 1.0, shape_lhs, &device).unwrap();
    let rhs = Tensor::randn(0.0f32, 1.0, shape_rhs, &device).unwrap();
    
    // 预热
    for _ in 0..3 {
        let _result = lhs.matmul(&rhs).unwrap();
    }
    
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE: 仅测量计算时间 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        let _result = lhs.matmul(&rhs).unwrap();
        durations.push(start.elapsed().as_secs_f64());
    }
    
    let mut sorted = durations.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mean = durations.iter().sum::<f64>() / durations.len() as f64;
    
    println!("  {}: Mean={:.3}ms, Median={:.3}ms", name, mean * 1000.0, median * 1000.0);
    
    (mean, median)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Candle 0.9.1 Matrix Multiplication Verification Test");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Configuration:");
    println!("  Device:  CPU");
    println!("  Samples: 10 per test");
    println!();
    
    println!("Running benchmarks...");
    println!();
    
    // General matmul tests (matching burn-bench)
    benchmark_matmul(1, 6144, 6144, 6144, "General [1, 6144, 6144] × [1, 6144, 6144]", 10);
    benchmark_matmul(2, 5000, 5000, 5000, "General [2, 5000, 5000] × [2, 5000, 5000]", 10);
    benchmark_matmul(4, 4096, 4096, 4096, "General [4, 4096, 4096] × [4, 4096, 4096]", 10);
    benchmark_matmul(8, 2048, 2048, 2048, "General [8, 2048, 2048] × [8, 2048, 2048]", 10);
    benchmark_matmul(16, 1024, 1024, 1024, "General [16, 1024, 1024] × [16, 1024, 1024]", 10);
    
    println!();
    println!("═══════════════════════════════════════════════════════════");
}
