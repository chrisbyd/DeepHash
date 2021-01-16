#!/bin/bash

lr=0.00005 
q_lambda=0.0
alpha=10.0 # 0.2
dataset=vehicleID # imagenet # cifar10,  nuswide_81, coco
log_dir=tflog
test_mode='800'

if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

export TF_CPP_MIN_LOG_LEVEL=3
#                                                         lr  output_dim  iter    q_lamb    alpha     dataset     gpu    log_dir
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 256     200000    $q_lambda  $alpha   $dataset    0   $log_dir  ../../data/vehicleID  $test_mode
