#!/bin/bash
SCENES="chair lego drums ficus hotdog materials mic ship"
DATA_ROOT=path/to/data/nerf_synthetic
# launch training jobs for all scenes.
for scene in $SCENES; do
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --config=configs/lego.txt \
    --datadir="$DATA_ROOT"/"$scene" \
    --expname=tensorf_"$scene"_VM_relu_black
done
