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
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 256     10000    $q_lambda  $alpha   $dataset    0   $log_dir  ../../data/VeRi  $test_mode | tee > VeRi_256.txt
echo "Starting training hash bit 512"
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 512     10000    $q_lambda  $alpha   $dataset    0   $log_dir  ../../data/VeRi  $test_mode | tee > VeRi_512.txt
#echo "Star training 1024"
#CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 1024     200000    $q_lambda  $alpha   $dataset    0   $log_dir  ../../data/vehicleID  $test_mode
#echo "Start training 2048"
#CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 2048    200000    $q_lambda  $alpha   $dataset    0   $log_dir  ../../data/vehicleID  $test_mode





