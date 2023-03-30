#!/bin/bash
DATA_ROOT=path/to/data/nerf_synthetic
ROOT_DIR=ckpt/nerf_synthetic
SCENES="lego chair drums ficus hotdog materials mic ship"
# launch training jobs for all scenes.
for scene in $SCENES; do
  python opt.py \
    "$DATA_ROOT"/"$scene" \
    --train_dir="$ROOT_DIR"/"$scene" \
    --config=configs/syn.json
done