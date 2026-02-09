# MFU（Model FLOPs Utilization）计算方法

MFU 表示训练时**实际达到的算力**占**硬件理论峰值算力**的比例，用于衡量训练对 GPU 的利用程度。

## 定义

```
MFU = 实际达到的 TFLOPS / (GPU 数量 × 单卡峰值 TFLOPS)
```

- **实际达到的 TFLOPS** = 每步的 FLOPs × 每秒步数（steps/s），再换算成 TFLOPs。
- **单卡峰值 TFLOPS**：取该 GPU 在 FP16（或当前训练精度）下的理论峰值，见下表。

## 本仓库中的实现

### 1. 每步 FLOPs（FLOPs/step）

训练一步包含一次前向和一次反向（反向约 2× 前向 FLOPs），因此：

```
FLOPs/step = 3 × FLOPs_forward_per_sample × global_batch_size
```

- **FLOPs_forward_per_sample**：单样本、单次前向的 FLOPs，由 Latte Transformer 结构估计。
- **global_batch_size** = `local_batch_size × world_size`（所有 GPU 上的总 batch）。

### 2. 单样本前向 FLOPs（Latte 估计）

Latte 为 Transformer，单层 FLOPs 近似为：

- **Attention**：`4·n·d² + 2·n²·d`（QKV 投影、注意力分数、输出投影等）
- **MLP**：约 `8·n·d²`

其中：

- `n` = 序列长度 = `num_frames × latent_size²`（如 16×32×32 = 16384）
- `d` = 隐藏维度（如 Latte-S/2 为 384）
- 总 FLOPs_forward = `num_layers × (每层 FLOPs)`

代码中见 `train.py` 的 `estimate_latte_flops_per_forward(num_frames, latent_size, hidden_size, num_layers)`。

### 3. 实际达到的 TFLOPS

```
achieved_TFLOPs = (FLOPs/step × steps_per_sec) / 10^12
```

即：每步算量 × 每秒步数，再换算成 TFLOPs。

### 4. MFU 公式

```
MFU = achieved_TFLOPs / (world_size × peak_TFLOPs_per_GPU)
```

- **world_size**：GPU 数量。
- **peak_TFLOPs_per_GPU**：单卡 FP16 理论峰值（TFLOPs），按 GPU 型号查表或由 `get_peak_tflops_fp16(device)` 给出。

若 MFU 接近 1（100%），表示训练算力接近硬件峰值；通常因内存带宽、通信等会低于 1。

## 常见 GPU 峰值（FP16，约值）

| GPU     | 峰值 TFLOPS (FP16) |
|---------|---------------------|
| 2080 Ti | 26.9                |
| 3080    | 35.6                |
| 3090    | 35.6                |
| 4090    | 82.6                |
| A100    | 312                 |
| H100    | 989                 |

## 日志中的输出

训练启动时（rank 0）会打印：

- **Parameters**：模型参数量（M）。
- **FLOPs/sample(fwd)**：单样本单次前向 TFLOPs。
- **FLOPs/step**：每步总 FLOPs（3×前向×global_batch），单位 TFLOPs。
- **GPU**：设备名与单卡峰值 TFLOPs。
- **MFU formula**：上述公式说明。

每隔 `log_every` 步会打印：

- **Loss / GradNorm / Steps/Sec**
- **Achieved**：当前实际达到的 TFLOPs/s。
- **MFU**：当前 MFU 百分比。

## 参考文献

- 概念与定义可参考 Chinchilla、Llama 等工作中对 MFU 的使用。
- 本仓库实现见 `train.py` 中 `get_peak_tflops_fp16`、`estimate_latte_flops_per_forward` 及日志与 tensorboard 中的 MFU 记录。
