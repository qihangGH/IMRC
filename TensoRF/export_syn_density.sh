#!/bin/bash
DATA_ROOT=path/to/data/nerf_synthetic
SCENES="chair lego drums ficus hotdog materials mic ship"
for scene in $SCENES; do
  CUDA_VISIBLE_DEVICES=0 python export_density.py \
    --config=configs/lego.txt \
    --datadir="$DATA_ROOT"/"$scene" \
    --reso=512,512,512 \
    --radius=1.5,1.5,1.5 \
    --ckpt=log/"$scene"/"$scene".th
done
