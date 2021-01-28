#!/bin/bash

lr=0.002
q_lambda=0.0001
subspace_num=4
dataset=vehicleID # cifar10, nuswide_81, vehicleID, VeRi
log_dir=tflog

if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

filename="lr_${lr}_cqlambda_${q_lambda}_subspace_num_${subspace_num}_T_${T}_K_${K}_graph_laplacian_lambda_${gl_lambda}_gl_loss_${gl_loss}_dataset_${dataset}"
model_file="models/${filename}.npy"
export TF_CPP_MIN_LOG_LEVEL=3
#                                                          lr  output  iter    q_lamb      n_sub   dataset     gpu    log_dir
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 256     8000    $q_lambda   4       $dataset    0 $log_dir ../../data/vehicleID | tee >vehicleID_256.txt
#                                                          lr  output  iter    q_lamb      n_sub   dataset     gpu    log_dir
echo "Start training 512"
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 512     8000    $q_lambda   4       $dataset    0 $log_dir ../../data/vehicleID | tee >vehicleID_512.txt
#echo "Start training 1024"
#CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 1024     8000    $q_lambda   4       $dataset    0 $log_dir ../../data/VeRi | tee >VeRi_1024.txt
#echo "Start training 2048"
#CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 2048     8000    $q_lambda   4       $dataset    0 $log_dir ../../data/VeRi | tee >VeRi_2048.txt
