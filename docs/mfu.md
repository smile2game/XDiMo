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

### 2. 单样本前向 FLOPs（Latte 结构估计）

Latte 为 patch-wise Transformer，序列长度为：

```
n = num_frames × (latent_size / patch_size)^2
```

其中：
- `latent_size = image_size / 8`（VAE 下采样 8 倍）
- `patch_size` 是 Latte 的 patch 大小（默认 2）

Latte 单层 FLOPs 近似为：

- **Attention**：`4·n·d² + 2·n²·d`（QKV 投影、注意力分数、输出投影等）
- **MLP**：约 `8·n·d²`  
  如果是 MoE（top_k>1），MLP 计算按 `top_k` 粗略放大：  
  `MLP_MoE ≈ 8·n·d²·top_k`

其中：

- `d` = 隐藏维度（如 Latte-S/2 为 384）
- 总 FLOPs_forward = `num_layers × (Attention + MLP)`

代码中见 `train.py` 的 `estimate_latte_flops_per_forward(...)`，已考虑 `patch_size` 和 `top_k`。

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
| H100    | 494                 |
| H200    | 494                 |

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

## 用模型结构/参数量推估 MFU（实用说明）

MFU 本质需要 **FLOPs/step**，而 FLOPs 来自结构与运行形状。参数量可以辅助近似，但不能完全确定 FLOPs（因为还与序列长度 `n`、patch_size、num_frames、top_k 等有关）。

### A. 结构法（推荐，和代码一致）

1. 计算 token 数：
```
n = num_frames × (latent_size / patch_size)^2
```

2. 单层 FLOPs：
```
F_layer ≈ (4·n·d^2 + 2·n^2·d) + 8·n·d^2·top_k
```

3. 单样本前向 FLOPs：
```
F_fwd ≈ num_layers × F_layer
```

4. 每步 FLOPs：
```
F_step ≈ 3 × F_fwd × global_batch_size
```

5. MFU：
```
MFU = (F_step × steps_per_sec / 1e12) / (world_size × peak_TFLOPs_per_GPU)
```

### B. 参数量法（粗略估计，需谨慎）

如果你只知道参数量 `P`，可以做一个**上界级别**的粗估：
- Transformer 中参数量大致与 `O(num_layers × d^2)` 成正比；
- FLOPs 与 `O(num_layers × n × d^2)` 成正比，还额外包含 `O(num_layers × n^2 × d)` 的注意力项；
- 因此仅靠 `P` 无法确定 FLOPs，必须结合 `n`（由 `num_frames`、`latent_size`、`patch_size` 决定）。

一个可用的近似是先用结构反推 `d, num_layers`（或从配置里读），再按 **结构法**计算 FLOPs。

结论：  
**最可靠的方法**是使用模型结构参数（`num_layers`、`hidden_size`、`patch_size`、`num_frames`、`top_k`）来计算 FLOPs，再结合吞吐得到 MFU；仅用参数量会有较大误差。

## 示例：Latte-L/2-MoE（F16S3，256²）的一次 MFU 计算

以常见配置为例（仅用于说明计算流程，数值可按实际配置替换）：

- `image_size=256`，因此 `latent_size = 256/8 = 32`
- `num_frames=16`
- `patch_size=2`
- `hidden_size=1024`
- `num_layers=24`
- `top_k=2`（MoE）
- 假设 `global_batch_size = 4`，`steps_per_sec = 1.2`
- 假设单卡峰值 `peak_TFLOPs=91.1`（6000ada），`world_size=1`

1. 序列长度：
```
n = num_frames × (latent_size / patch_size)^2
  = 16 × (32/2)^2
  = 16 × 16^2
  = 4096
```

2. 单层 FLOPs：
```
F_layer ≈ (4·n·d^2 + 2·n^2·d) + 8·n·d^2·top_k
       ≈ (4·4096·1024^2 + 2·4096^2·1024) + 8·4096·1024^2·2
```

3. 单样本前向 FLOPs：
```
F_fwd ≈ num_layers × F_layer
```

4. 每步 FLOPs：
```
F_step ≈ 3 × F_fwd × global_batch_size
```

5. 实际 TFLOPs 与 MFU：
```
achieved_TFLOPs = (F_step × steps_per_sec) / 1e12
MFU = achieved_TFLOPs / (world_size × peak_TFLOPs)
```

如果你需要我把上述数值完整算出具体 TFLOPs/MFU，我可以按你给的实测吞吐直接代入算出最终结果。
