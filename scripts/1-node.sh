#!/bin/bash
# 在 XDiMo 目录提交: cd XDiMo && jsub scripts/1-node.sh
#JSUB -J lhj-test
#JSUB -q gpu_6000ada
#JSUB -n 64
#JSUB -gpgpu 8
#JSUB -cwd /public/home/liuhuijie/dits/XDiMo
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

mkdir -p ./output/logs ./output/samples
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte

MASTER_NODE=$(echo "$JH_HOSTS" | awk '{print $1}')
echo "主节点: $MASTER_NODE"
# 单节点 GPU 数量：与 #JSUB -gpgpu 8 一致；若改 -gpgpu 需同步修改此处
GPUS_PER_NODE=8
echo "每个节点的GPU数量: $GPUS_PER_NODE"

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE train.py --config ./configs/ffs/ffs_train.yaml
