#!/bin/bash
# Latte-L/2-MoE 4卡 6000ada | 表一-A 4×6000ada
# 卡数由 -gpgpu 决定；-n = 卡数*8
#JSUB -J l2moe-6000ada-4gpu
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