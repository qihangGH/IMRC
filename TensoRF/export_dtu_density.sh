#!/bin/bash
DATA_ROOT=path/to/public_data
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
for scene in $SCENES; do
  CUDA_VISIBLE_DEVICES=0 python export_density.py \
    --config=configs/dtu.txt \
    --datadir="$DATA_ROOT"/dtu_scan"$scene" \
    --reso=512,512,512 \
    --radius=1.,1.,1. \
    --ckpt=log/tensorf_dtu_VM_relu_black_"$scene"/tensorf_dtu_VM_relu_black_"$scene".th
done
