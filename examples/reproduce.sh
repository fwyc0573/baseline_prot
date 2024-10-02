#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/local/ytyang/yichengfeng/Proteus
export NCCL_DEBUG=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1

GPU_NUM=64
# cluster=clusters/dgx1_v100_2ib/n8_g8.json
cluster=clusters/h800/n8_g8.json
PP=8
TP=4
# DP=$((${GPU_NUM}/${TP}/${PP}))

# bs=128
NUM_MICBATCH=1
MICRO_BATCH_SIZE=2
# GLOBAL_BATCH_SZIE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * DP))

# size variables
MODEL_SIZE=13 # "tiny"

if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=84;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

vocab_size=50257 #50257
MAX_SEQ_LEN=2048 #50257


USE_FLEXFLOW=true # true

if [ "$USE_FLEXFLOW" = true ]; then
    MODE="flexflow"
    FLEXFLOW_ARG="--flexflow"
else
    MODE="proteus"
    FLEXFLOW_ARG=""
fi

output_file="./log_test/${MODE}_GPU${GPU_NUM}_PP${PP}_TP${TP}_MICBATCH${NUM_MICBATCH}_BATCHSIZE${MICRO_BATCH_SIZE}_MODEL${MODEL_SIZE}.txt"

    # --reprofile \

## NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH \
mpirun -np 1 python megatron_gpt.py -model gpt \
    -global-bs 4 -n-macro-batch $NUM_MICBATCH \
    -nlayer $NUM_LAYERS -seq-length $MAX_SEQ_LEN -hidden-size $HIDDEN_SIZE -nheads $NUM_HEAD  -vocab-size $vocab_size \
    -ps pp -mp-deg $TP -pp-deg $PP -zero 0 --no-seq-first \
    -cluster $cluster \
    --profile-iters 3 \
    $FLEXFLOW_ARG | tee $output_file