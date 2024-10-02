python alexnet.py -model alexnet -bs 32 -cluster clusters/dgx1_v100_2ib/n8_g8.json -ps dp --profile-iters 1 #--reprofile
python alexnet.py -model alexnet -bs 32 -cluster clusters/dgx1_v100_2ib/n8_g8.json -ps pp --profile-iters 1
python alexnet.py -model alexnet -bs 32 -cluster clusters/dgx1_v100_2ib/n1_g8.json -ps dp --profile-iters 1

sleep 10s
python alexnet.py -model resnet50 -bs 32 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --profile-iters 50 --reprofile
sleep 10s
python alexnet.py -model inception_v3 -bs 32 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --profile-iters 50 --reprofile
sleep 10s
export CUDA_VISIBLE_DEVICES=5
python megatron_gpt.py -bs 4 -version layer -cluster clusters/dgx1_v100_2ib/n1_g1.json --bucket-size 35 --profile-iters 50 --reprofile


python alexnet.py -model vgg19 -bs 128 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --profile-iters 1 --reprofile
python alexnet.py -model vgg19 -bs 128 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --profile-iters 1 --reprofile --disable-collective


# bs = 2(per_gpu)*dp*2 = 4 * dp
python alexnet.py -model vgg19 -bs 16 -cluster clusters/dgx1_v100_2ib/n1_g4.json -ps pp --profile-iters 1 #--disable-collective



export PYTHONPATH=$PYTHONPATH:/local/ytyang/yichengfeng/Proteus
# global batchdp*num_mic_ba*mic_per_gpu = 4*dp = 32
# bs = args.bs * ndev


# groundtruth check: -bs 256  -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --bucket-size 25 --profile-iters 3 --flexflow
python alexnet.py -model vgg19 -bs 256 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --bucket-size 25 --profile-iters 3 --reprofile
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model resnet152 -bs 128 -cluster clusters/dgx1_v100_2ib/n1_g1.json -ps dp --bucket-size 1000000 --profile-iters 2 --reprofile



# VGG19 TEST ddp
export CUDA_VISIBLE_DEVICES=0
python alexnet.py -model vgg19 -bs 128 -cluster clusters/a100/n1_g8.json -ps dp --profile-iters 3 --flexflow --reprofile
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model vgg19 -bs 128 -cluster clusters/a100/n1_g8.json -ps dp --profile-iters 3 --reprofile

# VGG19 TEST pp=2
export CUDA_VISIBLE_DEVICES=0
python alexnet.py -model vgg19 -bs 64 -cluster clusters/a100/n1_g8.json -ps pp --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model vgg19 -bs 64 -cluster clusters/a100/n1_g8.json -ps pp --profile-iters 3


# VGG19 TEST pp=4
export CUDA_VISIBLE_DEVICES=0
python alexnet.py -model vgg19 -bs 32 -cluster clusters/a100/n1_g8.json -ps pp --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model vgg19 -bs 32 -cluster clusters/a100/n1_g8.json -ps pp --profile-iters 3

# -----------------------------------
# VGG19—_b256 128gpus with overlap
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model vgg19 -bs 256 -cluster clusters/a100_2ib/n16_g8.json -ps dp --bucket-size 25 --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=2
python alexnet.py -model vgg19 -bs 256 -cluster clusters/a100_2ib/n16_g8.json -ps dp --bucket-size 25 --profile-iters 3



# VGG19—_b256 64gpus with overlap
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model vgg19 -bs 256 -cluster clusters/a100_2ib/n8_g8.json -ps dp --bucket-size 25 --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=5
python alexnet.py -model vgg19 -bs 256 -cluster clusters/a100_2ib/n8_g8.json -ps dp --bucket-size 25 --profile-iters 3


# VGG19_b256 32gpus with overlap
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model vgg19 -bs 256 -cluster clusters/a100_2ib/n4_g8.json -ps dp --bucket-size 25 --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=2
python alexnet.py -model vgg19 -bs 256 -cluster clusters/a100_2ib/n4_g8.json -ps dp --bucket-size 25 --profile-iters 3


# -----------------------------------
# resnet152_b128 128gpus with overlap
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model resnet152 -bs 128 -cluster clusters/a100_2ib/n16_g8.json -ps dp --bucket-size 25 --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=2
python alexnet.py -model resnet152 -bs 128 -cluster clusters/a100_2ib/n16_g8.json -ps dp --bucket-size 25 --profile-iters 3


# resnet152_b128 64gpus with overlap
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model resnet152 -bs 128 -cluster clusters/a100_2ib/n8_g8.json -ps dp --bucket-size 25 --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=2
python alexnet.py -model resnet152 -bs 128 -cluster clusters/a100_2ib/n8_g8.json -ps dp --bucket-size 25 --profile-iters 3


# resnet152_b128 32gpus with overlap
export CUDA_VISIBLE_DEVICES=1
python alexnet.py -model resnet152 -bs 128 -cluster clusters/a100_2ib/n4_g8.json -ps dp --bucket-size 25 --profile-iters 3 --flexflow
export CUDA_VISIBLE_DEVICES=2
python alexnet.py -model resnet152 -bs 128 -cluster clusters/a100_2ib/n4_g8.json -ps dp --bucket-size 25 --profile-iters 3

# -----------------------------------