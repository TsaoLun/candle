use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax;
use std::time::Instant;

fn format_duration(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.3}s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{:.3}ms", ms)
    } else if ms >= 0.001 {
        format!("{:.3}µs", ms * 1000.0)
    } else {
        format!("{:.3}ns", ms * 1_000_000.0)
    }
}

fn benchmark_with_prepare<P, F>(
    name: &str,
    shape_info: &str,
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
    
    // 预热 (3次)
    for _ in 0..3 {
        execute().expect("warmup failed");
    }
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE 阶段：仅测量计算 ===
    let mut durations = Vec::new();
    for _ in 0..num_samples {
        let start = Instant::now();
        execute().expect("execute failed");
        durations.push(start.elapsed().as_secs_f64() * 1000.0); // 转换为 ms
    }
    
    let mut sorted = durations.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mean = durations.iter().sum::<f64>() / durations.len() as f64;
    
    println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
             name, shape_info, format_duration(median));
    println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
             "----", "----", "----", "----", "----", "----", "----");
    
    (mean, median)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Candle 0.9.1 Benchmark - 与 burn-bench 对比测试                                                                         ║");
    println!("║  Device: CPU  |  Samples: 10  |  Warmup: 3  |  Mode: prepare/execute                                                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
    
    println!("| Benchmark                 | Burn Version | Shapes                        | Feature    | Backend       | Device | Median    |");
    println!("|---------------------------|--------------|-------------------------------|------------|---------------|--------|-----------|");
    
    // ==================== 1. UNARY OPERATIONS ====================
    {
        let shape = (32, 512, 1024);
        let input = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "unary-f32",
            "(32, 512, 1024)",
            || Ok(()),
            || { let _ = input.tanh()?; Ok(()) },
            10
        );
    }
    
    // ==================== 2. BINARY OPERATIONS ====================
    {
        let shape = (512, 512, 1024);
        let lhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "binary-f32",
            "(512, 512, 1024)",
            || Ok(()),
            || { let _ = lhs.mul(&rhs)?; Ok(()) },
            10
        );
    }
    
    {
        let shape = (512, 512, 1024);
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        benchmark_with_prepare(
            "binary_scalar-f32",
            "(512, 512, 1024)",
            || Ok(()),
            || { let _ = (&t * 2.5)?; Ok(()) },
            10
        );
    }
    
    // ==================== 3. MATRIX MULTIPLICATION ====================
    // General matmul configurations
    let matmul_configs = [
        (1, 6144, 6144, 6144),
        (2, 5000, 5000, 5000),
        (4, 4096, 4096, 4096),
        (4, 2048, 2048, 2048),
        (8, 1024, 1024, 1024),
        (16, 512, 512, 512),
        (32, 256, 256, 256),
    ];
    
    for (b, m, n, k) in matmul_configs.iter() {
        let lhs = Tensor::randn(0.0f32, 1.0, (*b, *m, *k), &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (*b, *k, *n), &device)?;
        let shape_info = format!("({}, {}, {})", b, m, k);
        benchmark_with_prepare(
            "matmul-general-f32",
            &shape_info,
            || Ok(()),
            || { let _ = lhs.matmul(&rhs)?; Ok(()) },
            10
        );
    }
    
    // Inner product
    {
        let b = 1;
        let k = 8192;
        let lhs = Tensor::randn(0.0f32, 1.0, (b, 1, k), &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (b, k, 1), &device)?;
        benchmark_with_prepare(
            "matmul-inner-f32",
            &format!("[({}, 1, {})({}), {}, 1)]", b, k, b, k),
            || Ok(()),
            || { let _ = lhs.matmul(&rhs)?; Ok(()) },
            10
        );
    }
    
    // Mat @ Vec
    for b in [1, 2] {
        let m = 8192;
        let k = 8192;
        let lhs = Tensor::randn(0.0f32, 1.0, (b, m, k), &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (b, k, 1), &device)?;
        benchmark_with_prepare(
            "matmul-mat@vec-f32",
            &format!("[({}, {}, {})({}, {}, 1)]", b, m, k, b, k),
            || Ok(()),
            || { let _ = lhs.matmul(&rhs)?; Ok(()) },
            10
        );
    }
    
    // Outer product
    {
        let b = 1;
        let m = 16384;
        let n = 16384;
        let lhs = Tensor::randn(0.0f32, 1.0, (b, m, 1), &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (b, 1, n), &device)?;
        benchmark_with_prepare(
            "matmul-outer-f32",
            &format!("[({}, {}, 1)({}, 1, {})]", b, m, b, n),
            || Ok(()),
            || { let _ = lhs.matmul(&rhs)?; Ok(()) },
            10
        );
    }
    
    {
        let b = 4;
        let m = 8192;
        let n = 8192;
        let lhs = Tensor::randn(0.0f32, 1.0, (b, m, 1), &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (b, 1, n), &device)?;
        benchmark_with_prepare(
            "matmul-outer-f32",
            &format!("[({}, {}, 1)({}, 1, {})]", b, m, b, n),
            || Ok(()),
            || { let _ = lhs.matmul(&rhs)?; Ok(()) },
            10
        );
    }
    
    // Vec @ Mat
    for b in [1, 2] {
        let k = 8192;
        let n = 8192;
        let lhs = Tensor::randn(0.0f32, 1.0, (b, 1, k), &device)?;
        let rhs = Tensor::randn(0.0f32, 1.0, (b, k, n), &device)?;
        benchmark_with_prepare(
            "matmul-vec@mat-f32",
            &format!("[({}, 1, {})({}, {}, {})]", b, k, b, k, n),
            || Ok(()),
            || { let _ = lhs.matmul(&rhs)?; Ok(()) },
            10
        );
    }
    
    // ==================== 4. REDUCE OPERATIONS ====================
    let reduce_shape = (2048, 256, 64);
    
    for axis in 0..3 {
        let t = Tensor::randn(0.0f32, 1.0, reduce_shape, &device)?;
        let name = format!("reduce-argmin-{}-f32", axis);
        benchmark_with_prepare(
            &name,
            "(2048, 256, 64)",
            || Ok(()),
            || { let _ = t.argmin_keepdim(axis)?; Ok(()) },
            10
        );
    }
    
    for axis in 0..3 {
        let t = Tensor::randn(0.0f32, 1.0, reduce_shape, &device)?;
        let name = format!("reduce-argmin-{}-fused-f32", axis);
        let fused_t = ((((&t + 5.0)?.log())?.tanh())? * 3.0)?;
        benchmark_with_prepare(
            &name,
            "(2048, 256, 64)",
            || Ok(()),
            || { let _ = fused_t.argmin_keepdim(axis)?; Ok(()) },
            10
        );
    }
    
    for axis in 0..3 {
        let t = Tensor::randn(0.0f32, 1.0, reduce_shape, &device)?;
        let name = format!("reduce-sum-{}-f32", axis);
        benchmark_with_prepare(
            &name,
            "(2048, 256, 64)",
            || Ok(()),
            || { let _ = t.sum_keepdim(axis)?; Ok(()) },
            10
        );
    }
    
    for axis in 0..3 {
        let t = Tensor::randn(0.0f32, 1.0, reduce_shape, &device)?;
        let name = format!("reduce-sum-{}-fused-f32", axis);
        let fused_t = ((((&t + 5.0)?.log())?.tanh())? * 3.0)?;
        benchmark_with_prepare(
            &name,
            "(2048, 256, 64)",
            || Ok(()),
            || { let _ = fused_t.sum_keepdim(axis)?; Ok(()) },
            10
        );
    }
    
    {
        let t = Tensor::randn(0.0f32, 1.0, reduce_shape, &device)?;
        benchmark_with_prepare(
            "reduce-sum-full-f32",
            "(2048, 256, 64)",
            || Ok(()),
            || { let _ = t.sum_all()?; Ok(()) },
            10
        );
    }
    
    // ==================== 5. SOFTMAX ====================
    let softmax_shapes = [
        (2, 6144, 6144),
        (4, 4096, 4096),
        (8, 2048, 2048),
        (16, 1024, 1024),
        (256, 256, 256),
    ];
    
    for shape in softmax_shapes.iter() {
        // dim 0
        {
            let t = Tensor::randn(0.0f32, 1.0, *shape, &device)?;
            let shape_info = format!("({}, {}, {})", shape.0, shape.1, shape.2);
            benchmark_with_prepare(
                "softmax-0-f32",
                &shape_info,
                || Ok(()),
                || { let _ = softmax(&t, 0)?; Ok(()) },
                10
            );
        }
        // dim 1
        {
            let t = Tensor::randn(0.0f32, 1.0, *shape, &device)?;
            let shape_info = format!("({}, {}, {})", shape.0, shape.1, shape.2);
            benchmark_with_prepare(
                "softmax-1-f32",
                &shape_info,
                || Ok(()),
                || { let _ = softmax(&t, 1)?; Ok(()) },
                10
            );
        }
        // dim 2 (last dimension)
        {
            let t = Tensor::randn(0.0f32, 1.0, *shape, &device)?;
            let shape_info = format!("({}, {}, {})", shape.0, shape.1, shape.2);
            benchmark_with_prepare(
                "softmax-2-f32",
                &shape_info,
                || Ok(()),
                || { let _ = softmax(&t, D::Minus1)?; Ok(()) },
                10
            );
        }
    }
    
    println!("\n✅ 所有测试完成！可与 burn-bench 结果进行对比\n");
    
    Ok(())
}
