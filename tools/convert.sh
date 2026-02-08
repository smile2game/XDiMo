#!/bin/bash
#JSUB -J lhj-test
#JSUB -q gpu_6000ada
#JSUB -n 64
#JSUB -gpgpu 8
#JSUB -e ./%J.error
#JSUB -o ./%J.output

source activate latte

MASTER_NODE=$(echo "$JH_HOSTS" | awk '{print $1}')
echo "主节点: $MASTER_NODE"

GPUS_PER_NODE=$(echo $JH_GPU_RANK | cut -d';' -f1)
echo "每个节点的GPU数量: $GPUS_PER_NODE"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python convert_videos_to_frames.py \
  -s /public/home/liuhuijie/dits/Latte/test/1212 \
  -t /public/home/liuhuijie/dits/Latte/test/1212_frames \
  --video_ext mp4 \
  --target_size 256 \
  --force_fps 8 \
  --num_workers 8
