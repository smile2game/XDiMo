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

# 创建必要目录
mkdir -p ./output/logs ./output/samples
# 激活conda环境
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate latte

echo "=============================================="
echo "          H200 显卡可用性与运算测试             "
echo "=============================================="

# -----------------------------------------------------------------------
# 1. 基础信息
# -----------------------------------------------------------------------
echo ""
echo "===== 1. 驱动与设备信息 ====="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
  echo "  GPU $line"
done
nvidia-smi | grep -E "NVIDIA-SMI|Driver|CUDA"

# -----------------------------------------------------------------------
# 2. PyTorch 识别（调用独立Python脚本）
# -----------------------------------------------------------------------
echo ""
echo "===== 2. PyTorch CUDA 状态 ====="
# 写入独立 Python 脚本（用 printf 保证缩进为空格，避免 tab/空格混用导致 IndentationError）
printf '%s\n' \
  'import torch' \
  "print('  CUDA 可用:', torch.cuda.is_available())" \
  "print('  可见 GPU 数:', torch.cuda.device_count())" \
  'for i in range(torch.cuda.device_count()):' \
  '    props = torch.cuda.get_device_properties(i)' \
  "    print(f'  GPU {i}: {props.name}, 显存 {props.total_memory/1024**3:.1f} GB')" \
  > ./check_cuda.py
python ./check_cuda.py
rm -f ./check_cuda.py

# 检查显卡数量
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NGPU" -lt 4 ]; then
  echo "❌ 显卡数量不足 4 张 (当前 $NGPU)，跳过后续运算测试"
  echo "=============================================="
  exit 1
fi
echo "✅ 已识别 $NGPU 张显卡，继续运算测试"

# -----------------------------------------------------------------------
# 3. 单卡矩阵运算（调用独立Python脚本）
# -----------------------------------------------------------------------
# echo ""
# echo "===== 3. 单卡矩阵运算 (FP16, 8192×8192 矩阵乘) ====="
# cat > ./gpu_matmul_test.py << 'EOF'
# import torch
# import time

# dtype = torch.float16
# size = 8192
# warmup = 2
# repeat = 5

# for d in range(torch.cuda.device_count()):
#     torch.cuda.set_device(d)
#     torch.cuda.synchronize()
#     a = torch.randn(size, size, device='cuda', dtype=dtype)
#     b = torch.randn(size, size, device='cuda', dtype=dtype)
#     # 热身循环
#     for _ in range(warmup):
#         c = torch.matmul(a, b)
#     torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     # 测试循环
#     for _ in range(repeat):
#         c = torch.matmul(a, b)
#     torch.cuda.synchronize()
#     t1 = time.perf_counter()
#     elapsed = t1 - t0
#     gflops = 2 * (size**3) * repeat / elapsed / 1e9
#     checksum = c.float().sum().item()
#     print(f'  GPU {d}: {elapsed:.3f}s, {gflops:.0f} GFLOP/s, 校验和 {checksum:.2f}')
# EOF
# # 执行脚本
# python ./gpu_matmul_test.py
# # 删除临时脚本
# rm -f ./gpu_matmul_test.py

# -----------------------------------------------------------------------
# 4. 多卡并行（torchrun 启动，每卡做一次 all_reduce 验证通信）
# -----------------------------------------------------------------------
echo ""
echo "===== 4. 多卡 NCCL 通信测试 (torchrun 4 进程) ====="
NCCL_SCRIPT="./scripts/scaling_l2moe/h200/test_nccl.py"
NCCL_ERR=$(mktemp)
if torchrun --nnodes=1 --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) "$NCCL_SCRIPT" 2>"$NCCL_ERR"; then
  echo "  NCCL 通信测试通过"
else
  echo "  ⚠ NCCL 测试失败，报错如下："
  echo "----------------------------------------"
  cat "$NCCL_ERR"
  echo "----------------------------------------"
fi
rm -f "$NCCL_ERR"

# -----------------------------------------------------------------------
# 5. 小结
# -----------------------------------------------------------------------
echo ""
echo "=============================================="
echo "              测试完成                         "
echo "=============================================="

# 取消下面注释即可在测试通过后直接启动训练
# torchrun --nnodes=1 --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) \
#   train.py --config ./configs/ffs/ffs_train_bs1.yaml