#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
EPOCH=$8
BATCH_SIZE=$9
LR=${10}
HIDDEN_DIM=${11}
NODE_DIM=${12}
DR=${13}
READ_DIM=${14}
GRAPH_DIM=${15}
DATASET=${16}
FL_ALG=${17}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file
echo $NODE_DIM
mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 main_fedavg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --hidden_size $HIDDEN_DIM \
  --node_embedding_dim $NODE_DIM \
  --dropout $DR \
  --readout_hidden_dim $READ_DIM \
  --graph_embedding_dim $GRAPH_DIM \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --epochs $EPOCH \
  --fl_algorithm $FL_ALG \
  --batch_size $BATCH_SIZE \
  --lr $LR 
