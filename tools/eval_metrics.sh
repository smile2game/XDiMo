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
python /public/home/liuhuijie/dits/Latte/tools/calc_metrics_for_dataset.py \
--real_data_path //public/home/liuhuijie/dits/dataset/preprocess_ffs/train/images \
--fake_data_path /public/home/liuhuijie/dits/Latte/test/gen_frames \
--mirror 1 --gpus 8 --resolution 256 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0