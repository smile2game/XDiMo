#!/bin/bash
#JSUB -J lhj-test
#JSUB -q gpu_6000ada
#JSUB -n 64
#JSUB -R "span[ptile=32]"
#JSUB -gpgpu 8
#JSUB -e ./jlogs/%J.error 
#JSUB -o ./jlogs/%J.output

module load mpich/4.3.0b1-gcc-11.3.1
source activate latte

trap 'echo "强杀本用户所有python进程..."; pkill -9 -u $USER python; exit' SIGTERM SIGINT SIGHUP EXIT

MASTER_NODE=$(echo "$JH_HOSTS" | awk '{print $1}')
echo "主节点: $MASTER_NODE"
NUM_NODES=$(( $(echo "$JH_HOSTS" | wc -w) / 2 ))
echo "节点数量: $NUM_NODES"
GPUS_PER_NODE=$(echo $JH_GPU_RANK | cut -d';' -f1)
echo "每个节点的GPU数量: $GPUS_PER_NODE(仅适用于所有节点GPU数量一致)"
read -a NODELIST <<< "$JH_HOSTS"
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))
echo "总GPU数量为: $NUM_PROCESSES"
# export NCCL_DEBUG=INFO          # 启用NCCL调试日志
export OMP_NUM_THREADS=1        # 避免CPU线程竞争

for ((i=0; i<${#NODELIST[@]}; i+=2)); 
do
    mpirun --np 1 --host ${NODELIST[i]} \
        torchrun --nproc_per_node=$GPUS_PER_NODE \
                --nnodes=$NUM_NODES \
                --node_rank=$((i/2)) \
                --master_addr=$MASTER_NODE \
                --master_port=12345 \
                train.py \
                --config ./configs/ffs/ffs_train.yaml  &
done
echo "分布式训练启动状态: $?"

