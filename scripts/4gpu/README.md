# 4 卡单节点对比脚本（6000ada / pro6000 / H200）

## 提交前必看：资源与队列限制

- **提交前务必先运行 `jhosts`** 查看各队列可用卡数，再决定提交哪些任务。
- **H200**：本环境仅 **5 张卡**，同时只能跑 **1 个** 4 卡任务。
- **pro6000**：有 **1 张卡被占**，同时只能跑 **1 个** 4 卡任务。
- **6000ada**：8 张卡，可同时跑 2 个 4 卡任务。
- **最大等待任务数 = 1**：PEND 只能有 1 个，多提交会报「待提交任务数达上限」。建议分批提交，先跑一批再补交。

## 资源规则（脚本内）

- **卡数**：由 `#JSUB -gpgpu` 决定（本目录下均为 4 卡）。
- **CPU 核数**：`#JSUB -n` 必须为 **卡数×8**（4 卡即 `-n 32`）。  
  `-n` 过高会导致 CPU 资源不足，任务无法启动。

## 目录结构

```
4gpu/
├── 6000ada/     # 队列 gpu_6000ada，8 张卡，可并行 2 个 4 卡任务
│   ├── baseline.sh   # bs=1
│   └── bs2.sh        # bs=2（MFU 优化）
├── pro6000/     # 队列 gpu_pro6000，1 张被占，同时仅能 1 个 4 卡任务
│   ├── baseline.sh
│   └── bs2.sh
├── h200/        # 队列 gpu_h200，共 5 张卡，同时仅能 1 个 4 卡任务
│   ├── baseline.sh   # bs=1
│   └── bs4.sh        # bs=4（MFU 优化）
├── submit_compare.sh   # 一次提交 6 个（需先 jhosts，注意 PEND 上限 1）
├── submit_first_batch.sh  # 建议：先提交 4 个（2×6000ada + 1×h200 + 1×pro6000）
├── submit_remaining.sh   # 有空位时再提交剩余 2 个（h200 bs4 + pro6000 bs2）
├── submit_pro6000_only.sh
└── README.md
```

## 使用

```bash
cd /public/home/liuhuijie/dits/XDiMo

# 1. 提交前先看资源
jhosts

# 2. 建议分批提交（避免超过 PEND 上限）
bash scripts/4gpu/submit_first_batch.sh   # 4 个任务
# 等部分任务跑完后再：
bash scripts/4gpu/submit_remaining.sh     # 剩余 2 个

# 或一次提交 6 个（若 jhosts 显示资源充足且能接受 1 个 PEND）
bash scripts/4gpu/submit_compare.sh

# 查看任务 / 杀任务
jjobs
jctrl kill <JOBID>
```

单独提交：`jsub < scripts/4gpu/6000ada/baseline.sh`。  
日志：`./output/logs/%J.output`、`%J.error`。
