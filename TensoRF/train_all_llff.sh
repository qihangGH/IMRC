#!/bin/bash
DATA_ROOT=path/to/data/nerf_llff_data
SCENES="fern flower fortress horns leaves orchids room trex"
export CUDA_VISIBLE_DEVICES=0
# launch training jobs for all scenes.
for scene in $SCENES; do
  python train.py \
    --config=configs/flower.txt \
    --datadir="$DATA_ROOT"/"$scene" \
    --expname=tensorf_"$scene"_VM
done
