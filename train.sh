#!bin/bash
# Paths 
RAF_PATH=/home/jihyun/data/RAF
RESNET_PATH=/home/jihyun/code/eac/eac_puzzle/model/resnet50_ft_weight.pkl
SAVE_PATH=/home/jihyun/code/eac/eac_puzzle/
# Project name : change every experiments! 
# PROJECT=puzzle_baseline
PROJECT=PUZZLE
# PROJECT=PUZZLE_w2_2_nrce_lam5_30
GPU_ID=2 # gpu id 

# LABEL_PATH=noise04.txt
# /home/jihyun/code/eac/eac_puzzle/raf-basic/EmoLabel/noise04.txt
# hyperparams 
# LABEL_PATH=list_patition_label.txt
# LABEL_PATH=noise01.txt
# LABEL_PATH=noise02.txt
# LABEL_PATH=noise03.txt
# LABEL_PATH=noise04.txt
# LABEL_PATH=noise05.txt
# LABEL_PATH=noise06.txt
LABEL_PATH=noise07.txt

BATCH_SIZE=64
lam=5
LAM_ELR=3
BETA_ELR=0.5
# LEARNING_RATE=0.0001 0.7


python main_ori.py \
    --wandb=${PROJECT} \
    --raf_path=${RAF_PATH} \
    --resnet50_path=${RESNET_PATH} \
    --save_path=${SAVE_PATH} \
    --batch_size=$BATCH_SIZE \
    --gpu=$GPU_ID \
    --label_path=$LABEL_PATH \
    --lam=$lam \
    --lam_ELR=$LAM_ELR \
    --beta_ELR=$BETA_ELR
    # --learning_rat=$LEARNING_RATE

# python main.py --label_path 'noise01.txt' --gpu 0
# python main.py --label_path 'noise02.txt'
# python main.py --label_path 'noise03_plus1.txt'

