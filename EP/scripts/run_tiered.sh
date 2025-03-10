#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --tasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# nvidia-smi
export CUDA_VISIBLE_DEVICES=1

# source activate PL-CFE

cd ..
# Please go to exp_configs to modify the configs for pretraining and finetuning

# python3 trainval.py -e pretrain_en_tiered -sb ./logs/tiered/pretrain-tiered -d ../data  --dataset tieredimagenet

python3 trainval_un.py -e finetune_en_tiered -sb ./logs/tiered/finetune-tiered -d ../data --dataset tieredimagenet

# python3 test_score.py -sb ./logs/tiered/finetune-tiered -d ../data