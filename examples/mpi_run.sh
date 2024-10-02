#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/local/ytyang/yichengfeng/Proteus
export NCCL_DEBUG=DEBUG
vocab_size=3200 #50257
export CUDA_VISIBLE_DEVICES=1

GPU_NUM=64
mp_deg=4
pp_deg=8
dp_deg=$((${GPU_NUM}/${mp_deg}/${pp_deg}))
NUM_MICBATCH=1
MICRO_BATCH_SIZE=2
bs=$((NUM_MICBATCH * MICRO_BATCH_SIZE * dp_deg))

cluster=clusters/h800/n8_g8.json

MODEL_SIZE=13 # "tiny"

if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


# NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL
mpirun -np 1 python megatron_gpt.py -model gpt \
    -bs 1 -n-macro-batch $NUM_MICBATCH \
    -nlayer $NUM_LAYERS -seq-length 2048 -hidden-size $HIDDEN_SIZE -nheads $NUM_HEAD -vocab-size $vocab_size \
    -ps pp -mp-deg $mp_deg -pp-deg $pp_deg -zero 0 --no-seq-first \
    -cluster $cluster \
    --flexflow \
    --profile-iters 1 | tee output.txt

# megatron_node
#--bucket-size 19 #--checkpoint #--no-share-bandwidth #--no-seq-first #--reprofile

