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
export OMP_NUM_THREADS=1

MASTER_NODE=$(echo "$JH_HOSTS" | awk '{print $1}')
NUM_NODES=$(( $(echo "$JH_HOSTS" | wc -w) / 2 ))
GPUS_PER_NODE=8                                      # 改1：直接写死8，最保险
read -a NODELIST <<< "$JH_HOSTS"

for ((i=0; i<${#NODELIST[@]}; i+=2)); 
do
    mpirun --np 1 --host ${NODELIST[i]} bash -c "
        trap 'echo \"[$(hostname)] 强杀进程\"; pkill -9 -u $USER python' SIGTERM SIGINT EXIT
        torchrun --nproc_per_node=$GPUS_PER_NODE \
                 --nnodes=$NUM_NODES \
                 --node_rank=$((i/2)) \
                 --master_addr=$MASTER_NODE \
                 --master_port=12345 \
                 train.py --config ./configs/ffs/ffs_train.yaml
    " &
done

wait