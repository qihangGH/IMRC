#!/bin/bash
DATA_ROOT=path/to/public_data
ROOT_DIR=opt/ckpt
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
for scene in $SCENES; do
  (python calc_cd_others.py \
    "$ROOT_DIR"/dtu_"$scene"/ckpt.npz \
    "$DATA_ROOT"/dtu_scan"$scene" \
    "$scene" \
    path/to/dtu_dataset/evaluation/MVSData) &
done
wait
