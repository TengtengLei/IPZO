#!/bin/bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --dataset cifar10 \
    --random_mode Randint \
    --batch_size 64 \
    --val_batch_size 64 \
    --seed '0x123456789' \
    --output_dir ./randint \
    > logs/cifar10_randint_num5_freeze75_hflip.log 2>&1 &

echo "PID: $!"
