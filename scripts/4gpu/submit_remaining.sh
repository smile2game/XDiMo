#!/bin/bash
# 剩余 2 个任务：h200 bs4 + pro6000 bs2。请先 jhosts 确认有空位再提交（H200 仅5卡、pro6000 1张被占，各同时只能 1 个 4 卡任务；PEND 上限 1）
# 用法: cd XDiMo && bash scripts/4gpu/submit_remaining.sh

cd "$(dirname "$0")/../.." || exit 1

echo "提交剩余 2 个任务（h200 bs4 + pro6000 bs2）..."
jsub < scripts/4gpu/h200/bs4.sh
jsub < scripts/4gpu/pro6000/bs2.sh

echo "完成。jjobs 查看任务"
