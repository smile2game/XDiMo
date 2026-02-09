#!/bin/bash
# 提交: cd XDiMo && jsub < scripts/pro6000-1node.sh
# 查看任务: jjobs
# 杀任务: jctrl kill <id>
#JSUB -J lhj-test
#JSUB -q gpu_pro6000
#JSUB -n 64
#JSUB -gpgpu 8
#JSUB -cwd /public/home/liuhuijie/dits/XDiMo
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

mkdir -p ./output/logs ./output/samples
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte

GPUS_PER_NODE=8
torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=$((29500 + RANDOM % 1000)) train.py --config ./configs/ffs/ffs_train.yaml
