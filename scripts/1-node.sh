#!/bin/bash
# 请在 XDiMo 仓库根目录提交任务
#JSUB -J lhj-test
#JSUB -q gpu_6000ada
#JSUB -n 64
#JSUB -gpgpu 8
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

mkdir -p ./output/logs ./output/samples
source activate latte

MASTER_NODE=$(echo "$JH_HOSTS" | awk '{print $1}')
echo "主节点: $MASTER_NODE"
GPUS_PER_NODE=$(echo $JH_GPU_RANK | cut -d';' -f1)
echo "每个节点的GPU数量: $GPUS_PER_NODE"

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE train.py --config ./configs/ffs/ffs_train.yaml
