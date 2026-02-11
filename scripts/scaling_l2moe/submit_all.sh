#!/bin/bash
# 提交表一实验：6000ada 4 个（1/2/4/8 卡）+ H200 1 个（4 卡），共 5 个任务
# 用法: cd XDiMo && bash scripts/scaling_l2moe/submit_all.sh

cd "$(dirname "$0")/../.." || exit 1

echo "提交 Latte-L/2-MoE 加速比实验（5 个任务）..."
# jsub < scripts/scaling_l2moe/6000ada/1gpu.sh
# jsub < scripts/scaling_l2moe/6000ada/2gpu.sh
# jsub < scripts/scaling_l2moe/6000ada/4gpu.sh
# jsub < scripts/scaling_l2moe/6000ada/8gpu.sh
# jsub < scripts/scaling_l2moe/h200/4gpu.sh
jsub < scripts/scaling_l2moe/h200/2gpu.sh

echo "提交完成。jjobs 查看任务，日志见 output/logs/"
