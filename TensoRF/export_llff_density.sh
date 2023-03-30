#!/bin/bash
DATA_ROOT=path/to/data/nerf_llff_data
SCENES="fern flower fortress horns leaves orchids room trex"
for scene in $SCENES; do
  CUDA_VISIBLE_DEVICES=0 python export_density.py \
    --config=configs/fern.txt \
    --datadir="$DATA_ROOT"/"$scene" \
    --reso=598,665,400 \
    --radius=1.496031746031746,1.6613756613756614,1.0 \
    --ckpt=VM-48/ckpt/"$scene"/"$scene".th
done
