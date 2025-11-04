use candle_core::{Device, Tensor};
use std::time::Instant;

fn benchmark_binary_mul(shape: (usize, usize, usize), num_samples: usize) -> (f64, f64) {
    let device = Device::Cpu;
    
    // === PREPARE: 创建输入 (只做一次) ===
    let lhs = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
    let rhs = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
    
    // 预热
    for _ in 0..3 {
        let _result = lhs.mul(&rhs).unwrap();
    }
    
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE: 仅测量计算时间 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        let _result = lhs.mul(&rhs).unwrap();
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
    
    (mean, median)
}

fn benchmark_binary_scalar(shape: (usize, usize, usize), num_samples: usize) -> (f64, f64) {
    let device = Device::Cpu;
    let scalar = 2.5f64;
    
    // === PREPARE: 创建输入 (只做一次) ===
    let tensor = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
    
    // 预热
    for _ in 0..3 {
        let _result = (&tensor * scalar).unwrap();
    }
    
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE: 仅测量计算时间 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        let _result = (&tensor * scalar).unwrap();
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
    
    (mean, median)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Candle 0.9.1 Binary Operations Verification Test");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    
    // Binary multiply test (tensor * tensor)
    let shape = (512, 512, 1024);
    println!("Test 1: Binary Multiply (Tensor × Tensor)");
    println!("  Shape:   {:?}", shape);
    println!("  Device:  CPU");
    println!("  Samples: 10");
    println!();
    
    let (mean, median) = benchmark_binary_mul(shape, 10);
    println!("  Mean:   {:.3}ms", mean * 1000.0);
    println!("  Median: {:.3}ms", median * 1000.0);
    println!();
    
    // Binary scalar multiply test
    println!("─────────────────────────────────────────────────────────");
    println!("Test 2: Binary Scalar Multiply (Tensor × Scalar)");
    println!("  Shape:   {:?}", shape);
    println!("  Device:  CPU");
    println!("  Samples: 10");
    println!();
    
    let (mean, median) = benchmark_binary_scalar(shape, 10);
    println!("  Mean:   {:.3}ms", mean * 1000.0);
    println!("  Median: {:.3}ms", median * 1000.0);
    println!();
    
    println!("═══════════════════════════════════════════════════════════");
}
