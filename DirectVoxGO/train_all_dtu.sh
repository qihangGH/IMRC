#!/bin/bash
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
export CUDA_VISIBLE_DEVICES=0
# launch training jobs for all scenes.
for scene in $SCENES; do
    python run.py \
      --config=configs/dtu/dtu"$scene".py \
      --render_test
done