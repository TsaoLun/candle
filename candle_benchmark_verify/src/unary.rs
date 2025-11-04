use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // 与 benchmark 相同的形状: [32, 512, 1024]
    let shape = (32, 512, 1024);
    
    println!("═══════════════════════════════════════════════════════════");
    println!("  Candle 0.9.1 Unary (tanh) Performance Verification Test");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Configuration:");
    println!("  Shape:     {:?}", shape);
    println!("  Device:    CPU");
    println!("  Operation: tanh()");
    println!("  Samples:   10");
    println!("  Mode:      prepare/execute (matching burn-bench)");
    println!();
    
    // === PREPARE 阶段：创建输入 (只做一次) ===
    println!("Preparing input tensor...");
    let input = Tensor::randn(0.0f32, 1.0, shape, &device)?;
    
    // 预热 3 次 (与 burnbench 一致)
    println!("Warming up (3 iterations)...");
    for _ in 0..3 {
        let _result = input.tanh()?;
    }
    
    // 睡眠 1 秒 (与 burnbench 一致)
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // === EXECUTE 阶段：仅测量计算时间 ===
    println!("Running benchmark...");
    let mut durations = Vec::new();
    
    for i in 0..10 {
        // 只执行操作，不分配新张量
        let start = Instant::now();
        let _result = input.tanh()?;
        let duration = start.elapsed();
        
        durations.push(duration);
        println!("  Sample {:2}: {:7.3}ms", i + 1, duration.as_secs_f64() * 1000.0);
    }
    
    // 统计结果
    println!();
    println!("─────────────────────────────────────────────────────────");
    println!("Statistical Results:");
    println!("─────────────────────────────────────────────────────────");
    
    let mean = durations.iter().sum::<std::time::Duration>().as_secs_f64() / durations.len() as f64;
    
    let mut sorted_durations = durations.clone();
    sorted_durations.sort();
    let median = if sorted_durations.len() % 2 == 0 {
        let mid = sorted_durations.len() / 2;
        (sorted_durations[mid - 1].as_secs_f64() + sorted_durations[mid].as_secs_f64()) / 2.0
    } else {
        sorted_durations[sorted_durations.len() / 2].as_secs_f64()
    };
    
    let min = sorted_durations.first().unwrap().as_secs_f64();
    let max = sorted_durations.last().unwrap().as_secs_f64();
    
    let variance = durations.iter()
        .map(|d| {
            let diff = d.as_secs_f64() - mean;
            diff * diff
        })
        .sum::<f64>() / durations.len() as f64;
    
    let std_dev = variance.sqrt();
    
    println!("  Mean:     {:7.3}ms", mean * 1000.0);
    println!("  Median:   {:7.3}ms", median * 1000.0);
    println!("  Min:      {:7.3}ms", min * 1000.0);
    println!("  Max:      {:7.3}ms", max * 1000.0);
    println!("  Std Dev:  {:7.3}ms", std_dev * 1000.0);
    println!("  Variance: {:7.6}ms²", variance * 1_000_000.0);
    println!();
    println!("─────────────────────────────────────────────────────────");
    println!("Comparison with burn-bench result:");
    println!("─────────────────────────────────────────────────────────");
    println!("  burn-bench (candle-cpu):    103.162ms (Median)");
    println!("  Direct test (candle 0.9.1): {:7.3}ms (Median)", median * 1000.0);
    println!("  Difference:                 {:+7.3}ms ({:+.1}%)", 
             (median * 1000.0) - 103.162,
             ((median * 1000.0) - 103.162) / 103.162 * 100.0);
    println!();
    
    // 分析结果
    println!("─────────────────────────────────────────────────────────");
    println!("Analysis:");
    println!("─────────────────────────────────────────────────────────");
    
    let diff_percent = ((median * 1000.0) - 103.162).abs() / 103.162 * 100.0;
    
    if diff_percent < 5.0 {
        println!("✓ Results are VERY CLOSE (< 5% difference)");
        println!("  The benchmark framework is working correctly.");
    } else if diff_percent < 15.0 {
        println!("✓ Results are SIMILAR (< 15% difference)");
        println!("  The benchmark framework overhead is acceptable.");
    } else if diff_percent < 30.0 {
        println!("⚠ Results show MODERATE difference (< 30%)");
        println!("  This could be due to:");
        println!("  - System load variations");
        println!("  - Different Burn wrapper overhead");
        println!("  - Compilation differences");
    } else {
        println!("✗ Results show SIGNIFICANT difference (> 30%)");
        println!("  This may indicate:");
        println!("  - Different computation paths in Burn wrapper");
        println!("  - Measurement methodology differences");
        println!("  - Version mismatch or configuration issues");
    }
    
    println!();
    println!("Note: The first sample is often slower due to cold cache.");
    println!("Median is a more robust measure than mean for performance.");
    println!("═══════════════════════════════════════════════════════════");
    
    Ok(())
}
