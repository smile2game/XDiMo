# XDiMo 仓库结构说明

本仓库由 Latte 整理而来，目录约定如下。**所有脚本请在仓库根目录执行**（`cd /path/to/XDiMo` 后再运行）。

```
XDiMo/
├── configs/          # 配置文件（如 ffs_train.yaml, ffs_sample.yaml）
├── data/             # 训练数据（请将预处理数据放在此处，如 data/ffs/train/videos）
├── ckpts/            # 权重
│   ├── shared/       # 下载的共享权重（如 sd-vae-ft-ema、i3d_torchscript.pt）
│   └── trained/      # 本仓库训练得到的权重
├── output/           # 输出
│   ├── logs/         # 训练日志、任务 jlogs 等
│   └── samples/      # 采样生成的视频/图像
├── scripts/          # 脚本
│   ├── train/        # 训练脚本
│   ├── sample/       # 采样脚本
│   ├── slurm/        # Slurm 任务脚本
│   ├── 1-node.sh
│   ├── multi-node.sh
│   └── sample.sh
├── src/              # 源码（模型、数据集、扩散、工具）
│   ├── models/
│   ├── datasets/
│   ├── diffusion/
│   └── utils.py
├── sample/           # 采样入口（sample_ddp.py, sample.py, pipeline_latte.py 等）
├── train.py          # 训练入口
├── fvd.py            # FVD 评测
└── .gitignore
```

- **数据**：在 `configs/ffs/ffs_train.yaml` 中设置 `data_path: "./data/ffs/train/videos"`，将预处理数据放到 `data/` 下即可。
- **权重**：共享预训练权重放入 `ckpts/shared/`，训练得到的权重可放入 `ckpts/trained/`；训练时 checkpoint 会先写在 `output/<实验名>/checkpoints/`。
- **运行**：在仓库根目录执行 `python train.py --config ./configs/ffs/ffs_train.yaml` 或 `bash scripts/1-node.sh`。

原始 Latte 仓库未做修改，仍位于 `dits/Latte`。
