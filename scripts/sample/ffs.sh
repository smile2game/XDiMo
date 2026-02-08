#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python sample/sample.py \
--config ./configs/ffs/ffs_sample.yaml \
--ckpt ./share_ckpts/ffs.pt \
--save_video_path ./test