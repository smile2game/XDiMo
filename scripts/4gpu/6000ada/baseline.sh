#!/bin/bash
# 4卡 6000ada baseline (bs=1) | 提交: cd XDiMo && jsub < scripts/4gpu/6000ada/baseline.sh
# 卡数由 -gpgpu 决定；-n = 卡数*8（CPU核数），过高会导致 CPU 资源不足无法启动
#JSUB -J xdimo-6000ada-4gpu-bs1
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
  train.py --config ./configs/ffs/ffs_train_4gpu_bs1.yaml
