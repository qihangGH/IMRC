#!/bin/bash
DATA_ROOT=path/to/public_data
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python opt_metric.py \
    "$DATA_ROOT"/dtu_scan"$scene" \
    ckpt/dtu_"$scene"/ckpt.npz \
    --config=configs/dtu.json \
    --dataset_type=dtu \
    --high_fq
done
