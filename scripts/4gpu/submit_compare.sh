#!/bin/bash
# 提交 4 卡对比实验：3 baseline + 2 MFU优化（6000ada/pro6000 各加 bs2）
# 用法: cd XDiMo && bash scripts/4gpu/submit_compare.sh
# 查看: jjobs | 杀任务: jctrl kill <id>

cd "$(dirname "$0")/../.." || exit 1

echo "提交 6 个任务（4 卡/任务，6000ada/pro6000 可并行 2 个）..."
jsub < scripts/4gpu/6000ada/baseline.sh
jsub < scripts/4gpu/6000ada/bs2.sh
jsub < scripts/4gpu/h200/baseline.sh
jsub < scripts/4gpu/h200/bs4.sh
jsub < scripts/4gpu/pro6000/baseline.sh
jsub < scripts/4gpu/pro6000/bs2.sh

echo "提交完成，运行 jjobs 查看任务"
