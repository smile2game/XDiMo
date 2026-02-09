# XDiMo 仓库结构说明

本仓库由 Latte 整理而来，**所有脚本请在仓库根目录执行**（`cd /path/to/XDiMo` 后再运行）。

```
XDiMo/
├── configs/              # 配置文件
│   └── ffs/
│       ├── ffs_train.yaml
│       └── ffs_sample.yaml
├── docs/                 # 文档（除 README 外的 .md）
│   ├── STRUCTURE.md
│   ├── perf.md
│   ├── datasets_evaluation.md
│   └── latte_diffusers.md
├── sample/               # 采样入口
│   ├── sample.py
│   ├── sample_ddp.py
│   ├── pipeline_latte.py
│   └── sample_t2x.py
├── scripts/              # 脚本
│   ├── train/            # 训练脚本（ffs_train.sh 等）
│   ├── sample/           # 采样脚本（ffs_ddp.sh 等）
│   ├── slurm/            # Slurm 任务（超算）
│   ├── 1-node.sh
│   ├── multi-node.sh
│   └── setup_ckpts.sh
├── share_ckpts/          # 共享权重（VAE 等，需自行下载，见 README）
├── xdimo/                # 源码
│   ├── models/
│   ├── datasets/
│   ├── diffusion/
│   └── utils.py
├── tools/                # FVD 等评测工具
├── train.py              # 训练入口
├── local_train.sh        # 双卡 2080Ti 训练一键脚本
├── fvd.py                # FVD 评测入口
├── environment.yml
├── requirements.txt
└── README.md
```

- **数据**：在 `configs/ffs/ffs_train.yaml` 中设置 `data_path`，指向视频目录（如 `/data/preprocess_ffs/train/videos`）。
- **权重**：VAE 等放在 `share_ckpts/`（见 README）；训练 checkpoint 会写在 `output/<实验名>/checkpoints/`。
- **运行**：在仓库根目录执行 `bash local_train.sh` 或 `torchrun --nnodes=1 --nproc_per_node=2 train.py --config ./configs/ffs/ffs_train.yaml`。
