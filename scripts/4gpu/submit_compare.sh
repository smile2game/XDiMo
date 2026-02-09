#!/bin/bash
# 提交 4 卡对比实验：6 个任务（三种卡各 2 个：baseline + MFU 优化）
# 用法: cd XDiMo && bash scripts/4gpu/submit_compare.sh
# 注意: 提交前先 jhosts；H200 仅 5 张卡、pro6000 有 1 张被占，各只能跑 1 个 4 卡任务；最大等待任务数 1，建议用 submit_first_batch.sh + submit_remaining.sh 分批提交。

cd "$(dirname "$0")/../.." || exit 1

echo "请确认已运行 jhosts 查看资源（H200 仅5卡、pro6000 1张被占、PEND 上限1）。"
echo "提交 6 个任务..."
jsub < scripts/4gpu/6000ada/baseline.sh
jsub < scripts/4gpu/6000ada/bs2.sh
jsub < scripts/4gpu/h200/baseline.sh
jsub < scripts/4gpu/h200/bs4.sh
jsub < scripts/4gpu/pro6000/baseline.sh
jsub < scripts/4gpu/pro6000/bs2.sh

echo "提交完成，运行 jjobs 查看任务"
