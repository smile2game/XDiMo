#!/bin/bash
#JSUB -J l2moe-h200-1gpu
#JSUB -q gpu_h200
#JSUB -n 16
#JSUB -gpgpu 1
#JSUB -cwd /public/home/liuhuijie/dits/XDiMo
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

mkdir -p ./output/logs ./output/samples
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte

export CUDA_LAUNCH_BLOCKING=1 # 同步 CUDA 便于定位 kernel 报错（若稳定后可去掉以提速）
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((29500 + RANDOM % 1000)) \
  train.py --config ./configs/ffs/ffs_train_bs1.yaml
