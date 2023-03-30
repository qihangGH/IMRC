#!/bin/bash
#JaxNeRF
DATA_ROOT=path/to/public_data
ROOT_DIR=../../jaxnerf/tmp/dtu
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
    python opt_metric_from_other_models.py \
      "$DATA_ROOT"/dtu_scan"$scene" \
      "$ROOT_DIR"/dtu_"$scene"/density_volume_fine_512.npy \
      --config=configs/dtu.json \
      --basis_dim=9 \
      --dataset_type=dtu \
      --high_fq
done
