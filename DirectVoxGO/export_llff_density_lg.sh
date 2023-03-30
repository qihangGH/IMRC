#!/bin/bash
SCENES="fern flower fortress horns leaves orchids room trex"
for scene in $SCENES; do
    CUDA_VISIBLE_DEVICES=0 python export_density.py \
      --config=configs/llff/"$scene"_lg.py \
      --reso=598,665,400 \
      --radius=1.496031746031746,1.6613756613756614,1.0
done
