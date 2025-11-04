use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax;
use std::time::Instant;

fn benchmark_softmax(
    shape: (usize, usize, usize),
    _dim: usize,
    num_samples: usize,
) -> (f64, f64) {
    let device = Device::Cpu;
    
    // 预热
    for _ in 0..3 {
        let tensor = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
        let _result = softmax(&tensor, D::Minus1).unwrap();
    }
    
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // 测试
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let tensor = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
        
        let start = Instant::now();
        let _result = softmax(&tensor, D::Minus1).unwrap();
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
    println!("  Candle 0.9.1 Softmax Verification Test");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Configuration:");
    println!("  Device:  CPU");
    println!("  Samples: 10 per test");
    println!();
    
    let test_configs = [
        (2, 6144, 6144),
        (4, 4096, 4096),
        (8, 2048, 2048),
        (16, 1024, 1024),
        (256, 256, 256),
    ];
    
    println!("Running benchmarks...");
    println!();
    
    for (a, b, c) in test_configs.iter() {
        let shape = (*a, *b, *c);
        println!("Shape: {:?}", shape);
        
        for dim in 0..3 {
            let (mean, median) = benchmark_softmax(shape, dim, 10);
            println!("  Dim {}: Mean={:.3}ms, Median={:.3}ms", 
                     dim, mean * 1000.0, median * 1000.0);
        }
        println!();
    }
    
    println!("═══════════════════════════════════════════════════════════");
}
