#!/bin/bash

# export PYTHONPATH=$PYTHONPATH:/local/ytyang/yichengfeng/Proteus
# export NCCL_DEBUG=DEBUG
# vocab_size=3200 #50257

# mp_deg=2
# pp_deg=2
# # cluster=clusters/dgx1_v100_1ib/n4_g8.json
# # cluster=clusters/a100/n8_g8.json
# cluster=clusters/dgx1_v100_2ib/n1_g8.json
# bs=16

# NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL \
# mpirun -np 1 python megatron_gpt.py -model gpt \
#     -global-bs $bs -n-macro-batch 1\
#     -nlayer 40 -seq-length 2048 -hidden-size 5120 -nheads 32 -vocab-size $vocab_size \
#     -ps pp -mp-deg $mp_deg -pp-deg $pp_deg -zero 0 --no-seq-first \
#     -cluster $cluster \
#     --flexflow \
#     --profile-iters 1 | tee output.txt

# megatron_node
#--bucket-size 19 #--checkpoint #--no-share-bandwidth #--no-seq-first #--reprofile

export PYTHONPATH=$PYTHONPATH:/local/ytyang/yichengfeng/Proteus
export NCCL_DEBUG=DEBUG

GPU_NUM=8
cluster=clusters/a100/n1_g8.json
PP=1
TP=1
DP=$((${GPU_NUM}/${TP}/${PP}))

# bs=128
NUM_MICBATCH=1
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SZIE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * DP))

# size variables
MODEL_SIZE=13 # "tiny"

if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=80; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 135 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

vocab_size=3200 #50257
MAX_SEQ_LEN=2048 #50257


USE_FLEXFLOW=true

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
# export CUDA_VISIBLE_DEVICES=1
# mpirun -np 1 python megatron_gpt.py -model gpt \
#     -global-bs 2 -n-macro-batch $NUM_MICBATCH \
#     -nlayer $NUM_LAYERS -seq-length $MAX_SEQ_LEN -hidden-size $HIDDEN_SIZE -nheads $NUM_HEAD  -vocab-size $vocab_size \
#     -ps pp -mp-deg $TP -pp-deg $PP -zero 0 --no-seq-first \
#     -cluster $cluster \
#     --profile-iters 3 \
#     --reprofile \
#     $FLEXFLOW_ARG | tee $output_file



# gpt2_b16 128gpus with overlap
cluster=clusters/a100_2ib/n8_g8.json
export CUDA_VISIBLE_DEVICES=5
python megatron_gpt.py -model gpt \
    -bs 16 -n-macro-batch $NUM_MICBATCH \
    -nlayer $NUM_LAYERS -seq-length $MAX_SEQ_LEN -hidden-size $HIDDEN_SIZE -nheads $NUM_HEAD  -vocab-size $vocab_size \
    -ps dp -zero 0 --no-seq-first \
    -cluster $cluster \
    --profile-iters 3 \
    --bucket-size 25 \
    $FLEXFLOW_ARG | tee $output_file




#---------------------------------------------------------------------
# check groundtruth 
# seq 512
export PYTHONPATH=$PYTHONPATH:/local/ytyang/yichengfeng/Proteus
export CUDA_VISIBLE_DEVICES=1
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/dgx1_v100_2ib/n1_g1.json --bucket-size 250000000 --no-seq-first --profile-iters 2 --reprofile



# gpt2_b16 128gpus with overlap
export CUDA_VISIBLE_DEVICES=0
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/a100_2ib/n16_g8.json --bucket-size 25 --no-seq-first --profile-iters 5
export CUDA_VISIBLE_DEVICES=0
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/a100_2ib/n16_g8.json --bucket-size 25 --no-seq-first --profile-iters 5 --flexflow


# gpt2_b16 64gpus with overlap
export CUDA_VISIBLE_DEVICES=0
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/a100_2ib/n8_g8.json --bucket-size 25 --no-seq-first --profile-iters 5
export CUDA_VISIBLE_DEVICES=0
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/a100_2ib/n8_g8.json --bucket-size 25 --no-seq-first --profile-iters 5 --flexflow

# gpt2_b16 32gpus with overlap
export CUDA_VISIBLE_DEVICES=0
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/a100_2ib/n4_g8.json --bucket-size 25 --no-seq-first --profile-iters 3
export CUDA_VISIBLE_DEVICES=0
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/a100_2ib/n4_g8.json --bucket-size 25 --no-seq-first --profile-iters 3 --flexflow


# gpt2_b16 32gpus with overlap
export CUDA_VISIBLE_DEVICES=0
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/a100/n1_g8.json --bucket-size 25 --no-seq-first --profile-iters 3 --reprofile 
export CUDA_VISIBLE_DEVICES=5
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/dgx1_v100_2ib/n4_g8.json --bucket-size 25 --no-seq-first --profile-iters 3 --flexflow

export CUDA_VISIBLE_DEVICES=7
python megatron_gpt.py -bs 16 -model gpt-2 -cluster clusters/dgx1_v100_2ib/n1_g1.json --bucket-size 25 --no-seq-first --profile-iters 3 --reprofile