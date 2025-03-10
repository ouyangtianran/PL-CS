#!/bin/bash

nvidia-smi
#hostname

source activate PL-CFE
cd ../cfe
pretrain_path='./pretrain/imagenet-embedding.pth.tar'
dataset='miniimagenet'
# for s in 'train' 'test' 'val'
s ='train'
do
python cfe_encoding.py -p $pretrain_path -d $dataset -s $s
done