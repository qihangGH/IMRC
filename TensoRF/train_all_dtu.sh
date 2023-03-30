#!/bin/bash
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
DATA_ROOT=path/to/public_data
# launch training jobs for all scenes.
for scene in $SCENES; do
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --config=configs/dtu.txt \
    --expname=tensorf_dtu_VM_relu_black_"$scene" \
    --datadir="$DATA_ROOT"/dtu_scan"$scene"
done

