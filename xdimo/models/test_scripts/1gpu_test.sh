#!/bin/bash
# xdimo main 测试 | 6000ada 单卡 | 结果见 output/logs/%J.output 与本目录 1gpu_test.log
#JSUB -J xdimo-1gpu-test
#JSUB -q gpu_6000ada
#JSUB -n 8
#JSUB -gpgpu 1
#JSUB -cwd /public/home/liuhuijie/dits/XDiMo
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p ./output/logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte
cd /public/home/liuhuijie/dits/XDiMo

echo "=== nvidia-smi (before) ==="
nvidia-smi

python xdimo/models/xdimo.py 2>&1 | tee "$SCRIPT_DIR/1gpu_test.log"

echo "=== nvidia-smi (after) ==="
nvidia-smi
