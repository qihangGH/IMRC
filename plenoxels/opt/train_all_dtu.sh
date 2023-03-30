#!/bin/bash
DATA_ROOT=path/to/public_data
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
# launch training jobs for all scenes.
for scene in $SCENES; do
  python opt.py \
    "$DATA_ROOT"/dtu_scan"$scene" \
    --train_dir=ckpt/dtu_"$scene" \
    --config=configs/dtu.json
done
