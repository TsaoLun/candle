# Candle Benchmark Verification

这个项目用于直接使用 Candle 0.9.1 验证 burn-bench 中的性能测试结果。

## 项目结构

```
candle_benchmark_verify/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs      # 运行所有测试
    ├── unary.rs     # 一元运算测试 (tanh, gelu, relu, exp, log)
    ├── binary.rs    # 二元运算测试 (mul, add, scalar ops)
    ├── matmul.rs    # 矩阵乘法测试
    ├── softmax.rs   # Softmax 测试
    └── reduce.rs    # 归约运算测试 (sum, argmin)
```

## 使用方法

### 运行所有测试

```bash
cargo run --release --bin all
```

### 运行单个测试

```bash
# Unary 操作测试
cargo run --release --bin unary

# Binary 操作测试
cargo run --release --bin binary

# 矩阵乘法测试
cargo run --release --bin matmul

# Softmax 测试
cargo run --release --bin softmax

# 归约操作测试
cargo run --release --bin reduce
```

## 测试内容

### 1. Unary Operations
- **Shape**: `(32, 512, 1024)`
- **Operations**: tanh, gelu, relu, exp, log
- **对应 burn-bench**: `unary` benchmark

### 2. Binary Operations
- **Shape**: `(512, 512, 1024)`
- **Operations**: 
  - Tensor × Tensor (mul)
  - Tensor + Tensor (add)
  - Tensor × Scalar (mul_scalar)
- **对应 burn-bench**: `binary` benchmark

### 3. Matrix Multiplication
- **Configurations**:
  - `[2, 2048, 2048] × [2, 2048, 2048]`
  - `[4, 1024, 1024] × [4, 1024, 1024]`
  - `[8, 512, 512] × [8, 512, 512]`
- **对应 burn-bench**: `matmul` benchmark

### 4. Softmax
- **Shapes**: 
  - `(4, 4096, 4096)`
  - `(16, 1024, 1024)`
- **Dimensions**: 0, 1, 2
- **对应 burn-bench**: `softmax` benchmark

### 5. Reduce Operations
- **Shape**: `(2048, 256, 64)`
- **Operations**:
  - sum_dim (axis 0, 1, 2)
  - sum_all
  - argmin (axis 0, 1, 2)
- **对应 burn-bench**: `reduce` benchmark

## 测试方法

每个测试都遵循与 burn-bench 相同的方法：

1. **预热阶段**: 运行 3 次操作（不计时）
2. **等待**: 睡眠 1 秒（或 500ms）
3. **测试阶段**: 运行 10 次采样
4. **统计**: 计算 Mean 和 Median

## 输出格式

```
═══════════════════════════════════════════════════════════
  Candle 0.9.1 [Operation] Verification Test
═══════════════════════════════════════════════════════════

Configuration:
  Shape:   (...)
  Device:  CPU
  Samples: 10

Results:
  Mean:   XXX.XXXms
  Median: XXX.XXXms
```

## 对比 burn-bench 结果

运行测试后，可以将结果与 burn-bench 的输出进行对比：

```bash
# 在 burn-bench 根目录运行
cargo bb run --benches unary --backends candle-cpu

# 然后在这个目录运行
cargo run --release --bin unary
```

## 注意事项

1. **必须使用 --release 模式**：性能测试应在优化模式下运行
2. **系统负载**：确保系统负载较低以获得稳定结果
3. **多次运行**：建议运行多次取平均值
4. **CPU 差异**：不同 CPU 的性能会有差异

## 依赖版本

- `candle-core = "0.9.1"` - 与 burn-bench 中 candle 后端使用的版本一致

## 许可

与 burn-bench 项目保持一致
