#!/bin/bash
# 第一批：4 个任务（2×6000ada + 1×h200 + 1×pro6000），避免超过 PEND 上限 1
# 提交前请先运行 jhosts 查看资源
# 用法: cd XDiMo && bash scripts/4gpu/submit_first_batch.sh

cd "$(dirname "$0")/../.." || exit 1

echo "提交第一批 4 个任务（2×6000ada + 1×h200 + 1×pro6000）..."
jsub < scripts/4gpu/6000ada/baseline.sh
jsub < scripts/4gpu/6000ada/bs2.sh
jsub < scripts/4gpu/h200/baseline.sh
jsub < scripts/4gpu/pro6000/baseline.sh

echo "完成。有空位时再运行: bash scripts/4gpu/submit_remaining.sh"
jjobs
