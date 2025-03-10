#!/bin/bash

nvidia-smi
#hostname

conda activate
conda activate PL-CFE

pretrain_path='./scripts/omni/checkpoint_0199.pth.tar'
pretrain_path='./pretrain/omni-embedding.pth.tar'
dataset='omniglot'
for s in 'train' 'test' 'val'
do
python ./cfe/cfe_encoding.py -p $pretrain_path -d $dataset -s $s
done