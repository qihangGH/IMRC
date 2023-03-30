#!/bin/bash
# DVGO
DATA_ROOT=path/to/public_data
ROOT_DIR=../DirectVoxGO/logs/dtu
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
for scene in $SCENES; do
  (python calc_cd.py \
    "$ROOT_DIR"/dvgo_dtu"$scene"/density_volume_512.npy \
    "$DATA_ROOT"/dtu_scan"$scene" \
    "$scene" \
    path/to/dtu_dataset/evaluation/MVSData \
    --min_thresh=1 \
    --max_thresh=100
    ) &
done
wait