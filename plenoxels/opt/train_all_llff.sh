#!/bin/bash
DATA_ROOT=path/to/data/nerf_llff_data
ROOT_DIR=ckpt/llff
SCENES="fern flower fortress horns leaves orchids room trex"
# launch training jobs for all scenes.
for scene in $SCENES; do
  python opt.py \
    "$DATA_ROOT"/"$scene" \
    --train_dir="$ROOT_DIR"/"$scene" \
    --config=configs/llff.json
done