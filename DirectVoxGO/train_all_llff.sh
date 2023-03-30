#!/bin/bash
SCENES="fern flower fortress horns leaves orchids room trex"
# launch training jobs for all scenes.
for scene in $SCENES; do
    CUDA_VISIBLE_DEVICES=0 python run.py \
      --config=configs/llff/"$scene".py \
      --render_test
done
