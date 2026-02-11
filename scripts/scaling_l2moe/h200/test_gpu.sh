#!/bin/bash
# Latte-L/2-MoE 4卡 H200 | 表一-B 4×H200（今晚只跑 1 个四卡任务）
# 卡数由 -gpgpu 决定；-n = 卡数*8
#JSUB -J l2moe-h200-4gpu
#JSUB -q gpu_h200
#JSUB -n 32
#JSUB -gpgpu 4
#JSUB -cwd /public/home/liuhuijie/dits/XDiMo
#JSUB -e ./output/logs/%J.error
#JSUB -o ./output/logs/%J.output

mkdir -p ./output/logs ./output/samples
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte

# 极简显卡测试（新增，仅10行内）
echo "===== 显卡可用性测试 ====="
nvidia-smi | grep -E "NVIDIA-SMI|H200|GPU"  # 只显示核心显卡信息
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('可用卡数:', torch.cuda.device_count())"
if [ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -ge 4 ]; then
    echo "✅ 4张显卡已识别，测试通过"
else
    echo "❌ 显卡数量不足4张，测试失败"
fi
echo "===== 测试结束 ====="

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) \
#   train.py --config ./configs/ffs/ffs_train_bs1.yaml
# nvidia-smi