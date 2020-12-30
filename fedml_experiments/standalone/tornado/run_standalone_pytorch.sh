#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2

WORKER_NUM=$3

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

MODEL=$7

DISTRIBUTION=$8

LR=$9

OPT=$10

SEED=$11

GLOBAL_TOPOLOGY=$12

GROUP_TOPOLOGY=$13

GROUP_METHOD=$14

CHAIN_NUM=$15

GROUP_NUM=$16

GLOBAL_COMM_ROUND=$17

GROUP_COMM_ROUND=$18

EPOCH=$19

python3 ./main.py \
--gpu $GPU \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--batch_size $BATCH_SIZE \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--lr $LR \
--client_optimizer $OPT \
--seed $SEED \
--global_topology $GLOBAL_TOPOLOGY \
--group_topology $GROUP_TOPOLOGY \
--group_method $GROUP_METHOD \
--chain_num $CHAIN_NUM \
--group_num $GROUP_NUM \
--global_comm_round $GLOBAL_COMM_ROUND \
--group_comm_round $GROUP_COMM_ROUND \
--epochs $EPOCH \
--frequency_of_the_test 10
