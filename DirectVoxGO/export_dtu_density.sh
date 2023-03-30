#!/bin/bash
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
RESO="512"
for scene in $SCENES; do
    for reso in $RESO; do
      CUDA_VISIBLE_DEVICES=0 python export_density.py \
        --config=configs/dtu/dtu"$scene".py \
        --reso="$reso","$reso","$reso" \
        --radius=1.0,1.0,1.0
    done
done
