#!/bin/bash
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
export CUDA_VISIBLE_DEVICES=0
# launch training jobs for all scenes.
for scene in $SCENES; do
    python -m jaxnerf.train \
      --data_dir=path/to/public_data/dtu_scan"$scene" \
      --train_dir=jaxnerf/tmp/dtu/dtu_"$scene" \
      --config=jaxnerf/configs/dtu \
      --trans_coef=0.1
done
