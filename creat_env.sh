#!/bin/bash


conda create -y --name PL-CS python=3.9
source activate PL-CS

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install torchmeta==1.8.0
pip install -U scikit-learn
pip install haven-ai==0.6.7