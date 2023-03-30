#!/bin/bash
DATA_ROOT=path/to/public_data
ROOT_DIR=ckpt
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
for scene in $SCENES; do
    CUDA_VISIBLE_DEVICES=0 python visualize_res_color.py \
      "$DATA_ROOT"/dtu_scan"$scene" \
      "$ROOT_DIR"/dtu_"$scene"/color_res_vol_norm2_high_fq.npy \
      --config=configs/dtu.json \
      --color_scale=25
done
                                                                      #