#!/usr/bin/env python3
"""
多卡 NCCL 通信测试：torchrun 启动后每卡做一次 all_reduce，验证 4 卡通信正常。
用法: torchrun --nnodes=1 --nproc_per_node=4 --master_port=29500 test_nccl.py
"""
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # 每卡一个张量，all_reduce 求和
    x = torch.ones(1000, 1000, device=device, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = (1 + 2 + 3 + 4) * 1.0  # 4 卡时和为 10
    actual = x[0, 0].item()
    ok = abs(actual - expected) < 0.01
    print(f"  Rank {rank} (local_rank={local_rank}): all_reduce 结果={actual:.2f}, 期望≈{expected}, {'✅' if ok else '❌'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
