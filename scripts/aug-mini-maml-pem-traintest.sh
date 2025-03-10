#!/bin/bash
cd ..
out='./results/un_maml_pem/mini'

log_path=$(date "+%Y-%m-%d_%H%M%S")
mkdir -p "$out/$log_path"

# log_path='2023-12-08_143952'
# --resume_config "$out/$log_path/config.json"

nohup python train_test_aug.py ./data --dataset miniimagenet  --cuda-no 7 --K 500 --num-ways 5 --num-shots 1 --num-shots-test 5 --num-epochs 100  --n-warmup 10 --use-cuda --output-folder "$out/$log_path" --test-num-steps 50 --test-num-batches 125 --test-num-shots 1 >>"$out/$log_path/log.txt" & echo $! >>"$out/$log_path/pid.txt"