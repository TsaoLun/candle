use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "candle_benchmark_verify")]
#[command(about = "Candle 0.9.1 Benchmark - 对比 burn-bench", long_about = None)]
struct Args {
    /// 使用的后端设备 (cpu, metal)
    #[arg(long, default_value = "cpu")]
    device: String,
}

// 宏：自动处理 benchmark 错误，失败时跳过
macro_rules! run_benchmark {
    ($name:expr, $shape:expr, $device:expr, $prepare:expr, $execute:expr, $samples:expr) => {
        let _ = benchmark_with_prepare($name, $shape, $device, $prepare, $execute, $samples);
    };
}

fn get_device(device_name: &str) -> Result<Device, Box<dyn std::error::Error>> {
    match device_name.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        "metal" => {
            let device = Device::new_metal(0)?;
            Ok(device)
        }
        #[cfg(not(feature = "metal"))]
        "metal" => {
            Err("Metal backend not enabled. Compile with --features metal".into())
        }
        _ => Err(format!("Unsupported device: {}. Use 'cpu' or 'metal'", device_name).into()),
    }
}

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
    device: &Device,
    mut prepare: P,
    mut execute: F,
    num_samples: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>>
where
    P: FnMut() -> Result<(), Box<dyn std::error::Error>>,
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    // === PREPARE 阶段 ===
    if let Err(e) = prepare() {
        println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                 name, shape_info, "SKIPPED");
        println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                 "----", "----", "----", "----", "----", "----", "----");
        return Err(format!("prepare failed: {}", e).into());
    }
    
    // 预热 (3次)
    for i in 0..3 {
        if let Err(e) = execute() {
            println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                     name, shape_info, "SKIPPED");
            println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                     "----", "----", "----", "----", "----", "----", "----");
            return Err(format!("warmup #{} failed: {}", i+1, e).into());
        }
        // 确保 GPU 操作完成
        device.synchronize()?;
    }
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE 阶段：仅测量计算 ===
    let mut durations = Vec::new();
    for i in 0..num_samples {
        let start = Instant::now();
        if let Err(e) = execute() {
            println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                     name, shape_info, "SKIPPED");
            println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                     "----", "----", "----", "----", "----", "----", "----");
            return Err(format!("sample #{} failed: {}", i+1, e).into());
        }
        // 确保 GPU 操作完成后才停止计时
        device.synchronize()?;
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
    
    Ok((mean, median))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let device = get_device(&args.device)?;
    let device_display = args.device.to_uppercase();
    
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Candle 0.9.1 Benchmark - 与 burn-bench 对比测试                                                                         ║");
    println!("║  Device: {:<7}  |  Samples: 10  |  Warmup: 3  |  Mode: prepare/execute                                                ║", device_display);
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
    
    println!("| Benchmark                 | Burn Version | Shapes                        | Feature    | Backend       | Device | Median    |");
    println!("|---------------------------|--------------|-------------------------------|------------|---------------|--------|-----------|");
    
    // ==================== 1. UNARY OPERATIONS ====================
    {
        let shape = (32, 512, 1024);
        if let Ok(input) = Tensor::randn(0.0f32, 1.0, shape, &device) {
            run_benchmark!(
                "unary-f32",
                "(32, 512, 1024)",
                &device,
                || Ok(()),
                || { let _ = input.tanh()?; Ok(()) },
                10
            );
        } else {
            println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                     "unary-f32", "(32, 512, 1024)", "SKIPPED");
            println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                     "----", "----", "----", "----", "----", "----", "----");
        }
    }
    
    // ==================== 2. BINARY OPERATIONS ====================
    if let Ok(lhs) = Tensor::randn(0.0f32, 1.0, (512, 512, 1024), &device) {
        if let Ok(rhs) = Tensor::randn(0.0f32, 1.0, (512, 512, 1024), &device) {
            run_benchmark!(
                "binary-f32",
                "(512, 512, 1024)",
                &device,
                || Ok(()),
                || { let _ = lhs.mul(&rhs)?; Ok(()) },
                10
            );
        }
        
        run_benchmark!(
            "binary_scalar-f32",
            "(512, 512, 1024)",
            &device,
            || Ok(()),
            || { let _ = (&lhs * 2.5)?; Ok(()) },
            10
        );
    } else {
        println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                 "binary-*", "(512, 512, 1024)", "SKIPPED");
        println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                 "----", "----", "----", "----", "----", "----", "----");
    }
    
    // ==================== 3. MATRIX MULTIPLICATION ====================
    // Test if matmul works with a small example first
    let matmul_works = {
        let test_lhs = Tensor::randn(0.0f32, 1.0, (1, 4, 4), &device);
        let test_rhs = Tensor::randn(0.0f32, 1.0, (1, 4, 4), &device);
        match (test_lhs, test_rhs) {
            (Ok(l), Ok(r)) => l.matmul(&r).is_ok(),
            _ => false,
        }
    };
    
    if matmul_works {
        // 只保留小型快速测试，移除耗时超过2s的大型测试
        let matmul_configs = [
            // 移除: (1, 6144, 6144, 6144),  // ~1s
            // 移除: (2, 5000, 5000, 5000),  // ~1.2s
            // 移除: (4, 4096, 4096, 4096),  // ~1.3s
            (4, 2048, 2048, 2048),     // ~180ms
            (8, 1024, 1024, 1024),     // ~48ms
            (16, 512, 512, 512),       // ~11ms
            (32, 256, 256, 256),       // ~5ms
        ];
        
        for (b, m, n, k) in matmul_configs.iter() {
            if let Ok(lhs) = Tensor::randn(0.0f32, 1.0, (*b, *m, *k), &device) {
                if let Ok(rhs) = Tensor::randn(0.0f32, 1.0, (*b, *k, *n), &device) {
                    let shape_info = format!("({}, {}, {})", b, m, k);
                    run_benchmark!(
                        "matmul-general-f32",
                        &shape_info,
                        &device,
                        || Ok(()),
                        || { let _ = lhs.matmul(&rhs)?; Ok(()) },
                        10
                    );
                }
            }
        }
        
        // Inner product
        {
            let b = 1;
            let k = 8192;
            if let Ok(lhs) = Tensor::randn(0.0f32, 1.0, (b, 1, k), &device) {
                if let Ok(rhs) = Tensor::randn(0.0f32, 1.0, (b, k, 1), &device) {
                    run_benchmark!(
                        "matmul-inner-f32",
                        &format!("[({}, 1, {})({}), {}, 1)]", b, k, b, k),
                        &device,
                        || Ok(()),
                        || { let _ = lhs.matmul(&rhs)?; Ok(()) },
                        10
                    );
                }
            }
        }
        
        // Mat @ Vec
        for b in [1, 2] {
            let m = 8192;
            let k = 8192;
            if let Ok(lhs) = Tensor::randn(0.0f32, 1.0, (b, m, k), &device) {
                if let Ok(rhs) = Tensor::randn(0.0f32, 1.0, (b, k, 1), &device) {
                    run_benchmark!(
                        "matmul-mat@vec-f32",
                        &format!("[({}, {}, {})({}, {}, 1)]", b, m, k, b, k),
                        &device,
                        || Ok(()),
                        || { let _ = lhs.matmul(&rhs)?; Ok(()) },
                        10
                    );
                }
            }
        }
        
        // Outer product
        for (b, m, n) in [(1, 16384, 16384), (4, 8192, 8192)] {
            if let Ok(lhs) = Tensor::randn(0.0f32, 1.0, (b, m, 1), &device) {
                if let Ok(rhs) = Tensor::randn(0.0f32, 1.0, (b, 1, n), &device) {
                    run_benchmark!(
                        "matmul-outer-f32",
                        &format!("[({}, {}, 1)({}, 1, {})]", b, m, b, n),
                        &device,
                        || Ok(()),
                        || { let _ = lhs.matmul(&rhs)?; Ok(()) },
                        10
                    );
                }
            }
        }
        
        // Vec @ Mat
        for b in [1, 2] {
            let k = 8192;
            let n = 8192;
            if let Ok(lhs) = Tensor::randn(0.0f32, 1.0, (b, 1, k), &device) {
                if let Ok(rhs) = Tensor::randn(0.0f32, 1.0, (b, k, n), &device) {
                    run_benchmark!(
                        "matmul-vec@mat-f32",
                        &format!("[({}, 1, {})({}, {}, {})]", b, k, b, k, n),
                        &device,
                        || Ok(()),
                        || { let _ = lhs.matmul(&rhs)?; Ok(()) },
                        10
                    );
                }
            }
        }
    } else {
        println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                 "matmul-*", "various", "SKIPPED");
        println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                 "----", "----", "----", "----", "----", "----", "----");
    }
    
    // ==================== 4. REDUCE OPERATIONS ====================
    let reduce_shape = (2048, 256, 64);
    
    // Test if reduce works first
    let reduce_works = {
        let test_t = Tensor::randn(0.0f32, 1.0, (8, 8, 8), &device);
        match test_t {
            Ok(t) => t.sum_keepdim(0).is_ok() && t.argmin_keepdim(0).is_ok(),
            _ => false,
        }
    };
    
    if reduce_works {
        for axis in 0..3 {
            if let Ok(t) = Tensor::randn(0.0f32, 1.0, reduce_shape, &device) {
                let name = format!("reduce-argmin-{}-f32", axis);
                run_benchmark!(
                    &name,
                    "(2048, 256, 64)",
                    &device,
                    || Ok(()),
                    || { let _ = t.argmin_keepdim(axis)?; Ok(()) },
                    10
                );
            }
        }
        
        for axis in 0..3 {
            if let Ok(t) = Tensor::randn(0.0f32, 1.0, reduce_shape, &device) {
                if let Ok(fused_t) = ((((&t + 5.0).and_then(|x| x.log())).and_then(|x| x.tanh())).and_then(|x| x * 3.0)) {
                    let name = format!("reduce-argmin-{}-fused-f32", axis);
                    run_benchmark!(
                        &name,
                        "(2048, 256, 64)",
                        &device,
                        || Ok(()),
                        || { let _ = fused_t.argmin_keepdim(axis)?; Ok(()) },
                        10
                    );
                }
            }
        }
        
        for axis in 0..3 {
            if let Ok(t) = Tensor::randn(0.0f32, 1.0, reduce_shape, &device) {
                let name = format!("reduce-sum-{}-f32", axis);
                run_benchmark!(
                    &name,
                    "(2048, 256, 64)",
                    &device,
                    || Ok(()),
                    || { let _ = t.sum_keepdim(axis)?; Ok(()) },
                    10
                );
            }
        }
        
        for axis in 0..3 {
            if let Ok(t) = Tensor::randn(0.0f32, 1.0, reduce_shape, &device) {
                if let Ok(fused_t) = ((((&t + 5.0).and_then(|x| x.log())).and_then(|x| x.tanh())).and_then(|x| x * 3.0)) {
                    let name = format!("reduce-sum-{}-fused-f32", axis);
                    run_benchmark!(
                        &name,
                        "(2048, 256, 64)",
                        &device,
                        || Ok(()),
                        || { let _ = fused_t.sum_keepdim(axis)?; Ok(()) },
                        10
                    );
                }
            }
        }
        
        if let Ok(t) = Tensor::randn(0.0f32, 1.0, reduce_shape, &device) {
            run_benchmark!(
                "reduce-sum-full-f32",
                "(2048, 256, 64)",
                &device,
                || Ok(()),
                || { let _ = t.sum_all()?; Ok(()) },
                10
            );
        }
    } else {
        println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                 "reduce-*", "(2048, 256, 64)", "SKIPPED");
        println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                 "----", "----", "----", "----", "----", "----", "----");
    }
    
    // ==================== 5. SOFTMAX ====================
    // 只保留小型快速测试，移除耗时超过2s的大型测试
    let softmax_shapes = [
        // 移除: (2, 6144, 6144),   // dim0: ~2.3s, dim1: ~3s
        // 移除: (4, 4096, 4096),   // dim0: ~1.8s, dim1: ~2.5s
        (8, 2048, 2048),      // dim0: ~856ms, dim1: ~1.1s, dim2: ~342ms
        (16, 1024, 1024),     // dim0: ~423ms, dim1: ~519ms, dim2: ~170ms
        (256, 256, 256),      // dim0: ~413ms, dim1: ~506ms, dim2: ~171ms
    ];
    
    // Test if softmax works first
    let softmax_works = {
        let test_t = Tensor::randn(0.0f32, 1.0, (4, 4, 4), &device);
        match test_t {
            Ok(t) => softmax(&t, 0).is_ok(),
            _ => false,
        }
    };
    
    if softmax_works {
        for shape in softmax_shapes.iter() {
            // dim 0
            if let Ok(t) = Tensor::randn(0.0f32, 1.0, *shape, &device) {
                let shape_info = format!("({}, {}, {})", shape.0, shape.1, shape.2);
                run_benchmark!(
                    "softmax-0-f32",
                    &shape_info,
                    &device,
                    || Ok(()),
                    || { let _ = softmax(&t, 0)?; Ok(()) },
                    10
                );
            }
            // dim 1
            if let Ok(t) = Tensor::randn(0.0f32, 1.0, *shape, &device) {
                let shape_info = format!("({}, {}, {})", shape.0, shape.1, shape.2);
                run_benchmark!(
                    "softmax-1-f32",
                    &shape_info,
                    &device,
                    || Ok(()),
                    || { let _ = softmax(&t, 1)?; Ok(()) },
                    10
                );
            }
            // dim 2 (last dimension)
            if let Ok(t) = Tensor::randn(0.0f32, 1.0, *shape, &device) {
                let shape_info = format!("({}, {}, {})", shape.0, shape.1, shape.2);
                run_benchmark!(
                    "softmax-2-f32",
                    &shape_info,
                    &device,
                    || Ok(()),
                    || { let _ = softmax(&t, D::Minus1)?; Ok(()) },
                    10
                );
            }
        }
    } else {
        println!("| {:<25} | main         | {:<29} | candle-cpu | `candle<cpu>` | Cpu    | {:>9} |", 
                 "softmax-*", "various", "SKIPPED");
        println!("| {:<25} | {:<12} | {:<29} | {:<10} | {:<13} | {:<6} | {:<9} |",
                 "----", "----", "----", "----", "----", "----", "----");
    }
    
    println!("\n✅ 所有测试完成！可与 burn-bench 结果进行对比\n");
    
    Ok(())
}
