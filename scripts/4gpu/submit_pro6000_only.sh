#!/bin/bash
# 仅提交 pro6000 的 2 个任务。注意：pro6000 有 1 张卡被占，同时只能跑 1 个 4 卡任务，且 PEND 上限 1，建议先 jhosts 再提交，或一次只提交 1 个。
# 用法: cd XDiMo && bash scripts/4gpu/submit_pro6000_only.sh

cd "$(dirname "$0")/../.." || exit 1

echo "提交 pro6000 的 2 个任务（同时仅能跑 1 个，请先 jhosts）..."
jsub < scripts/4gpu/pro6000/baseline.sh
jsub < scripts/4gpu/pro6000/bs2.sh
echo "完成。jjobs 查看任务。"
