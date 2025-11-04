use candle_core::{Device, Tensor};
use std::time::Instant;

fn benchmark_sum_dim(shape: (usize, usize, usize), axis: usize, num_samples: usize) -> (f64, f64) {
    let device = Device::Cpu;
    
    // === PREPARE: 创建输入 (只做一次) ===
    let tensor = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
    
    // 预热
    for _ in 0..3 {
        let _result = tensor.sum_keepdim(axis).unwrap();
    }
    
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE: 仅测量计算时间 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        let _result = tensor.sum_keepdim(axis).unwrap();
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

fn benchmark_sum_full(shape: (usize, usize, usize), num_samples: usize) -> (f64, f64) {
    let device = Device::Cpu;
    
    // === PREPARE: 创建输入 (只做一次) ===
    let tensor = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
    
    // 预热
    for _ in 0..3 {
        let _result = tensor.sum_all().unwrap();
    }
    
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE: 仅测量计算时间 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        let _result = tensor.sum_all().unwrap();
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

fn benchmark_argmin(shape: (usize, usize, usize), axis: usize, num_samples: usize) -> (f64, f64) {
    let device = Device::Cpu;
    
    // === PREPARE: 创建输入 (只做一次) ===
    let tensor = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
    
    // 预热
    for _ in 0..3 {
        let _result = tensor.argmin_keepdim(axis).unwrap();
    }
    
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE: 仅测量计算时间 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        let _result = tensor.argmin_keepdim(axis).unwrap();
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
    println!("  Candle 0.9.1 Reduce Operations Verification Test");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Configuration:");
    println!("  Shape:   (2048, 256, 64)");
    println!("  Device:  CPU");
    println!("  Samples: 10 per test");
    println!();
    
    let shape = (2048, 256, 64);
    
    println!("Test 1: Sum along dimensions");
    for axis in 0..3 {
        let (mean, median) = benchmark_sum_dim(shape, axis, 10);
        println!("  Axis {}: Mean={:.3}ms, Median={:.3}ms", 
                 axis, mean * 1000.0, median * 1000.0);
    }
    println!();
    
    println!("Test 2: Full sum");
    let (mean, median) = benchmark_sum_full(shape, 10);
    println!("  Full:  Mean={:.3}ms, Median={:.3}ms", mean * 1000.0, median * 1000.0);
    println!();
    
    println!("Test 3: ArgMin along dimensions");
    for axis in 0..3 {
        let (mean, median) = benchmark_argmin(shape, axis, 10);
        println!("  Axis {}: Mean={:.3}ms, Median={:.3}ms", 
                 axis, mean * 1000.0, median * 1000.0);
    }
    println!();
    
    println!("═══════════════════════════════════════════════════════════");
}
