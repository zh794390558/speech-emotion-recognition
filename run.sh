#!/bin/bash

set -x 
stage=$1
DATA_PATH=/nfs/project/zhanghui/data/emotion/iemocap/


if [[ $stage -le 0 ]];then
  python zscore.py --dataset_path  $DATA_PATH --feature_size 40 || exit 1
  python ExtractMel.py --dataset_path $DATA_PATH --feature_size 40 || exit 1
fi

if [[ $stage -le 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 python train.py --attention True --num_epochs 10 || exit 1
fi

if [[ $stage -le 2 ]]; then
  CUDA_VISIBLE_DEVICES=0 python eval.py --attention True || exit 1
fi
