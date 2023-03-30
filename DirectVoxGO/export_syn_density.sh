#!/bin/bash
SCENES="chair lego drums ficus hotdog materials mic ship"
for scene in $SCENES; do
    CUDA_VISIBLE_DEVICES=0 python export_density.py \
      --config=configs/nerf/"$scene".py \
      --reso=512,512,512 \
      --radius=1.5,1.5,1.5
done
