# Candle Benchmark Verification

这个项目用于直接使用 Candle 0.9.1 验证 burn-bench 中的性能测试结果。

cargo run --release --features metal --bin all

对比 `cargo bb run -b unary binary matmul softmax reduce -B candle-cpu`
