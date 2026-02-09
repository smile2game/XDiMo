#!/bin/bash
# 8卡 pro6000 | 提交: cd XDiMo && jsub < scripts/8gpu/pro6000.sh
#JSUB -J xdimo-pro6000-8gpu
#JSUB -q gpu_pro6000
#JSUB -n 64
#JSUB -gpgpu 8
#JSUB -cwd /public/home/liuhuijie/dits/XDiMo
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

mkdir -p ./output/logs ./output/samples
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte

torchrun --nnodes=1 --nproc_per_node=8 --master_port=$((29500 + RANDOM % 1000)) \
  train.py --config ./configs/ffs/ffs_train.yaml
