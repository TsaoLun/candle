use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax;
use std::time::Instant;

fn print_header(title: &str) {
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  {}", title);
    println!("═══════════════════════════════════════════════════════════\n");
}

fn print_separator() {
    println!("─────────────────────────────────────────────────────────");
}

fn benchmark_op<F>(
    name: &str,
    _shape: (usize, usize, usize),
    mut op: F,
    num_samples: usize,
) -> (f64, f64)
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    // 预热
    for _ in 0..3 {
        let _ = op();
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    // 测试
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        let _ = op();
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
    
    println!("  {:30} Mean={:7.3}ms  Median={:7.3}ms", 
             name, mean * 1000.0, median * 1000.0);
    
    (mean, median)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    print_header("Candle 0.9.1 Comprehensive Benchmark Verification");
    println!("Device: CPU");
    println!("Samples per test: 10");
    
    // 1. Unary Operations
    print_separator();
    println!("\n1. UNARY OPERATIONS\n");
    
    let shape = (32, 512, 1024);
    println!("Shape: {:?}\n", shape);
    
    benchmark_op("tanh", shape, || {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = t.tanh()?;
        Ok(())
    }, 10);
    
    benchmark_op("gelu", shape, || {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = t.gelu()?;
        Ok(())
    }, 10);
    
    benchmark_op("relu", shape, || {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = t.relu()?;
        Ok(())
    }, 10);
    
    benchmark_op("exp", shape, || {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = t.exp()?;
        Ok(())
    }, 10);
    
    benchmark_op("log", shape, || {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?.abs()?;
        let _ = t.log()?;
        Ok(())
    }, 10);
    
    // 2. Binary Operations
    print_separator();
    println!("\n2. BINARY OPERATIONS\n");
    
    let shape = (512, 512, 1024);
    println!("Shape: {:?}\n", shape);
    
    benchmark_op("mul (tensor × tensor)", shape, || {
        let lhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = lhs.mul(&rhs)?;
        Ok(())
    }, 10);
    
    benchmark_op("add (tensor + tensor)", shape, || {
        let lhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = lhs.add(&rhs)?;
        Ok(())
    }, 10);
    
    benchmark_op("mul_scalar (tensor × 2.5)", shape, || {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = (&t * 2.5)?;
        Ok(())
    }, 10);
    
    // 3. Matrix Multiplication
    print_separator();
    println!("\n3. MATRIX MULTIPLICATION\n");
    
    let configs = [
        (2, 2048, 2048, 2048),
        (4, 1024, 1024, 1024),
        (8, 512, 512, 512),
    ];
    
    for (b, m, n, k) in configs.iter() {
        let name = format!("[{}, {}, {}] × [{}, {}, {}]", b, m, k, b, k, n);
        benchmark_op(&name, (*b, *m, *k), || {
            let lhs = Tensor::randn(0.0f32, 1.0, (*b, *m, *k), &device)?;
            let rhs = Tensor::randn(0.0f32, 1.0, (*b, *k, *n), &device)?;
            let _ = lhs.matmul(&rhs)?;
            Ok(())
        }, 10);
    }
    
    // 4. Softmax
    print_separator();
    println!("\n4. SOFTMAX\n");
    
    let shapes = [(4, 4096, 4096), (16, 1024, 1024)];
    
    for shape in shapes.iter() {
        println!("\nShape: {:?}", shape);
        for dim in 0..3 {
            let name = format!("  dim={}", dim);
            benchmark_op(&name, *shape, || {
                let t = Tensor::randn(0.0f32, 1.0, *shape, &device)?;
                let _ = softmax(&t, D::Minus1)?;
                Ok(())
            }, 10);
        }
    }
    
    // 5. Reduce Operations
    print_separator();
    println!("\n5. REDUCE OPERATIONS\n");
    
    let shape = (2048, 256, 64);
    println!("Shape: {:?}\n", shape);
    
    for axis in 0..3 {
        let name = format!("sum_dim({})", axis);
        benchmark_op(&name, shape, || {
            let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
            let _ = t.sum_keepdim(axis)?;
            Ok(())
        }, 10);
    }
    
    benchmark_op("sum_all", shape, || {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let _ = t.sum_all()?;
        Ok(())
    }, 10);
    
    for axis in 0..3 {
        let name = format!("argmin({})", axis);
        benchmark_op(&name, shape, || {
            let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
            let _ = t.argmin_keepdim(axis)?;
            Ok(())
        }, 10);
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All benchmarks completed!");
    println!("═══════════════════════════════════════════════════════════\n");
    
    Ok(())
}
