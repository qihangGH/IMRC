#!/bin/bash
# JaxNeRF
DATA_ROOT=path/to/public_data
ROOT_DIR=../jaxnerf/tmp
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
# launch training jobs for all scenes.
for scene in $SCENES; do
  (python calc_cd.py \
    "$ROOT_DIR"/dtu_"$scene"/density_volume_fine.npy \
    "$DATA_ROOT"/dtu_scan"$scene" \
    "$scene" \
    path/to/dtu_dataset/evaluation/MVSData) &
done
wait
