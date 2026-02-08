#!/bin/bash
#JSUB -J lhj-test
#JSUB -q gpu_6000ada
#JSUB -n 64
#JSUB -gpgpu 8
#JSUB -e ./jlogs/%J.error
#JSUB -o ./jlogs/%J.output

source activate latte

MASTER_NODE=$(echo "$JH_HOSTS" | awk '{print $1}')
echo "主节点: $MASTER_NODE"

GPUS_PER_NODE=$(echo $JH_GPU_RANK | cut -d';' -f1)
echo "每个节点的GPU数量: $GPUS_PER_NODE"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 请在 XDiMo 仓库根目录执行，或取消下一行注释以自动 cd 到仓库根
# cd "$(dirname "$0")/.."
torchrun --nnodes=1 --nproc_per_node=8 sample/sample_ddp.py \
  --config ./configs/ffs/ffs_sample.yaml \
  --ckpt ./ckpts/trained/010-Latte-L-2-MoE-F16S3-ffs-Gc-Amp/0270000.pt \
  --save_video_path ./output/samples