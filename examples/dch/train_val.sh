#!/bin/bash

lr=0.00005
q_lambda=0.0
alpha=10.0 # 0.2
dataset=VeRi # imagenet # cifar10,  nuswide_81, coco, vehicledID
log_dir=tflog
test_mode='800'

if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

export TF_CPP_MIN_LOG_LEVEL=3
#                                                         lr  output_dim  iter    q_lamb    alpha     dataset     gpu    log_dir
echo "Starting training hash bit 256"
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     --output-dim 256 --batch-size 800 --dataset $dataset --iter-num 8000 |tee > VeRi_256.txt
echo "Starting training hash bit 512"
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py  --output-dim 512 --batch-size 800 --dataset $dataset --iter-num 8000 | tee > VeRi_512.txt
#echo "Star training 1024"
#CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     --output-dim 1024 --batch-size 800 --dataset $dataset --iter-num 8000 |tee > VeRi_1024.txt
#echo "Start training 2048"
#CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py     --output-dim 2048 --batch-size 800 --dataset $dataset --iter-num 8000 |tee > VeRi_2048.txt





