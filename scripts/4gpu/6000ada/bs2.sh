#!/bin/bash
# 4卡 6000ada MFU优化 (bs=2) | 提交: cd XDiMo && jsub < scripts/4gpu/6000ada/bs2.sh
#JSUB -J xdimo-6000ada-4gpu-bs2
#JSUB -q gpu_6000ada
#JSUB -n 32
#JSUB -gpgpu 4
#JSUB -cwd /public/home/liuhuijie/dits/XDiMo
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

mkdir -p ./output/logs ./output/samples
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte

torchrun --nnodes=1 --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) \
  train.py --config ./configs/ffs/ffs_train_4gpu_bs2.yaml
