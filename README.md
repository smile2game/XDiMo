# XDiMo

基于 [Latte](https://github.com/Vchitect/Latte) 整理的视频生成训练与评测仓库，适配本地双卡（如 2×2080Ti）与超算 Slurm 环境。

- **训练**：DDP 多卡、混合精度、梯度检查点，支持 FFS 等数据集。
- **采样**：单卡 / DDP 生成视频，可导出 .npz 用于 FVD。
- **评测**：FVD 等指标，见 `tools/` 与 [docs/datasets_evaluation.md](docs/datasets_evaluation.md)。

---

## 文件结构

```
XDiMo/
├── configs/ffs/          # FFS 训练/采样配置
├── docs/                 # 文档（STRUCTURE、数据集与评测、diffusers 等）
├── sample/               # 采样脚本（sample_ddp.py 等）
├── scripts/              # 训练/采样/slurm 脚本
├── share_ckpts/          # 共享权重（VAE，需自行下载）
├── xdimo/                # 模型、数据集、扩散、工具
├── tools/                # FVD 等评测
├── train.py              # 训练入口
├── local_train.sh        # 双卡 2080Ti 训练
└── fvd.py                # FVD 评测
```

详见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

---

## 环境

```bash
cd XDiMo
conda env create -f environment.yml
conda activate latte
# 或 pip install -r requirements.txt
```

需准备：

- **共享权重**：将 VAE（如 [sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-mse)）放到 `share_ckpts/vae/`（含 `config.json`、`diffusion_pytorch_model.bin`）。或从 [Latte 共享权重](https://huggingface.co/maxin-cn/Latte/tree/main) 按目录放到 `share_ckpts/`。
- **数据**：FFS 视频目录，例如 `/data/preprocess_ffs/train/videos` 下放 `.avi` 等视频文件。

---

## 在 2×2080Ti 上训练

在仓库根目录执行：

```bash
bash local_train.sh
```

即：

```bash
torchrun --nnodes=1 --nproc_per_node=2 train.py --config ./configs/ffs/ffs_train.yaml
```

- 配置：`configs/ffs/ffs_train.yaml`（默认 Latte-S/2、`data_path`、`pretrained_model_path: "./share_ckpts"`、`results_dir: "./output"`）。
- 修改数据路径：把 `data_path` 改为你的视频目录（如 `/data/preprocess_ffs/train/videos`）。
- 输出：日志与 checkpoint 在 `output/<实验名>/`，如 `output/000-Latte-S-2-F16S3-ffs-Gc-Amp/checkpoints/0050000.pt`。

---

## 在超算上训练（Slurm）

1. 将仓库同步到超算，配置好 conda 与 `share_ckpts`、数据路径。
2. 在 `configs/ffs/ffs_train.yaml` 中把 `data_path`、`pretrained_model_path` 等改为超算上的路径。
3. 使用 `scripts/slurm/ffs.slurm`（按集群修改 `#SBATCH` 分区、队列等）：

```bash
cd XDiMo
sbatch scripts/slurm/ffs.slurm
```

4. 或使用 `scripts/train/ffs_train.sh` 的 torchrun 命令，在作业脚本里调用（见 `scripts/1-node.sh`、`scripts/multi-node.sh` 参考）。

---

## 采样

使用训练得到的 checkpoint（如 S/2）生成视频。

**单卡**（示例）：

```bash
python sample/sample.py --config ./configs/ffs/ffs_sample.yaml --ckpt ./output/xxx/checkpoints/0050000.pt --save_video_path ./output/samples
```

**双卡 DDP**（生成更多视频、便于算 FVD）：

```bash
torchrun --nnodes=1 --nproc_per_node=2 sample/sample_ddp.py \
  --config ./configs/ffs/ffs_sample.yaml \
  --ckpt ./output/xxx/checkpoints/0050000.pt \
  --save_video_path ./output/samples
```

- `configs/ffs/ffs_sample.yaml` 中 `model`、`num_frames`、`frame_interval` 等需与训练配置一致；`pretrained_model_path` 指向 `./share_ckpts`（VAE）。
- 生成结果在 `./output/samples`（或你指定的路径），DDP 会额外生成 `.npz` 用于评测。
- 若仅做快速验证，可将 `configs/ffs/ffs_sample.yaml` 中 `num_fvd_samples` 调小（如 8），否则默认 2048 会较耗时。

---

## 验证精度（FVD 等）

1. 按 [docs/datasets_evaluation.md](docs/datasets_evaluation.md) 准备真实/生成视频（同分辨率、center-crop-resize 等）。
2. 使用 `tools/` 下脚本计算 FVD 等指标，例如：

```bash
cd XDiMo
bash tools/eval_metrics.sh
```

或使用仓库根目录的 `fvd.py`（若已对接）。具体命令与数据格式见 `tools/` 内 README 与 `docs/datasets_evaluation.md`。

---

## 参考

- Latte 论文与代码：[Latte](https://github.com/Vchitect/Latte)、[Project Page](https://maxin-cn.github.io/latte_project/)。
- 数据集与评测说明：本仓库 [docs/datasets_evaluation.md](docs/datasets_evaluation.md)、[docs/latte_diffusers.md](docs/latte_diffusers.md)。
- 性能与规模笔记：[docs/perf.md](docs/perf.md)。

## License

见 [LICENSE](LICENSE)。
