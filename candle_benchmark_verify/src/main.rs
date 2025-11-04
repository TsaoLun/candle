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

fn benchmark_with_prepare<P, F>(
    name: &str,
    mut prepare: P,
    mut execute: F,
    num_samples: usize,
) -> (f64, f64)
where
    P: FnMut() -> Result<(), Box<dyn std::error::Error>>,
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    // === PREPARE 阶段 ===
    prepare().expect("prepare failed");
    
    // 预热
    for _ in 0..3 {
        execute().expect("warmup failed");
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    // === EXECUTE 阶段：仅测量计算 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        execute().expect("execute failed");
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
    
    print_header("Candle 0.9.1 Comprehensive Benchmark (prepare/execute mode)");
    println!("Device: CPU");
    println!("Samples per test: 10");
    println!("Mode: Matching burn-bench (input allocated once, reused)");
    
    // 1. Unary Operations
    print_separator();
    println!("\n1. UNARY OPERATIONS\n");
    
    let shape = (32, 512, 1024);
    println!("Shape: {:?}\n", shape);
    
    {
        let input = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "tanh",
            || Ok(()),
            || { let _ = input.tanh()?; Ok(()) },
            10
        );
    }
    
    {
        let input = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "gelu",
            || Ok(()),
            || { let _ = input.gelu()?; Ok(()) },
            10
        );
    }
    
    {
        let input = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "relu",
            || Ok(()),
            || { let _ = input.relu()?; Ok(()) },
            10
        );
    }
    
    {
        let input = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "exp",
            || Ok(()),
            || { let _ = input.exp()?; Ok(()) },
            10
        );
    }
    
    {
        let input = Tensor::randn(0.0f32, 1.0, shape, &device)?.abs()?;
        benchmark_with_prepare(
            "log",
            || Ok(()),
            || { let _ = input.log()?; Ok(()) },
            10
        );
    }
    
    // 2. Binary Operations
    print_separator();
    println!("\n2. BINARY OPERATIONS\n");
    
    let shape = (512, 512, 1024);
    println!("Shape: {:?}\n", shape);
    
    {
        let lhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "mul (tensor × tensor)",
            || Ok(()),
            || { let _ = lhs.mul(&rhs)?; Ok(()) },
            10
        );
    }
    
    {
        let lhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "add (tensor + tensor)",
            || Ok(()),
            || { let _ = lhs.add(&rhs)?; Ok(()) },
            10
        );
    }
    
    {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "mul_scalar (tensor × 2.5)",
            || Ok(()),
            || { let _ = (&t * 2.5)?; Ok(()) },
            10
        );
    }
    
    // 3. Matrix Multiplication
    print_separator();
    println!("\n3. MATRIX MULTIPLICATION\n");
    
    let configs = [
        (2, 2048, 2048, 2048),
        (4, 1024, 1024, 1024),
        (8, 512, 512, 512),
    ];
    
    for (b, m, n, k) in configs.iter() {
        let lhs = Tensor::randn(0.0f32, 1.0, (*b, *m, *k), &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (*b, *k, *n), &device)?;
        let name = format!("[{}, {}, {}] × [{}, {}, {}]", b, m, k, b, k, n);
        benchmark_with_prepare(
            &name,
            || Ok(()),
            || { let _ = lhs.matmul(&rhs)?; Ok(()) },
            10
        );
    }
    
    // 4. Softmax
    print_separator();
    println!("\n4. SOFTMAX\n");
    
    let shapes = [(4, 4096, 4096), (16, 1024, 1024)];
    
    for shape in shapes.iter() {
        println!("\nShape: {:?}", shape);
        let t = Tensor::randn(0.0f32, 1.0, *shape, &device)?;
        let name = format!("  dim=-1");
        benchmark_with_prepare(
            &name,
            || Ok(()),
            || { let _ = softmax(&t, D::Minus1)?; Ok(()) },
            10
        );
    }
    
    // 5. Reduce Operations
    print_separator();
    println!("\n5. REDUCE OPERATIONS\n");
    
    let shape = (2048, 256, 64);
    println!("Shape: {:?}\n", shape);
    
    for axis in 0..3 {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let name = format!("sum_dim({})", axis);
        benchmark_with_prepare(
            &name,
            || Ok(()),
            || { let _ = t.sum_keepdim(axis)?; Ok(()) },
            10
        );
    }
    
    {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "sum_all",
            || Ok(()),
            || { let _ = t.sum_all()?; Ok(()) },
            10
        );
    }
    
    for axis in 0..3 {
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let name = format!("argmin({})", axis);
        benchmark_with_prepare(
            &name,
            || Ok(()),
            || { let _ = t.argmin_keepdim(axis)?; Ok(()) },
            10
        );
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All benchmarks completed!");
    println!("═══════════════════════════════════════════════════════════\n");
    
    Ok(())
}
