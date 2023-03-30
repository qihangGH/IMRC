#!/bin/bash
# TensoRF
DATA_ROOT=path/to/data/nerf_synthetic
ROOT_DIR=../../TensoRF/log
SCENES="chair lego drums ficus hotdog materials mic ship"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python opt_metric_from_other_models.py \
    "$DATA_ROOT"/"$scene" \
    "$ROOT_DIR"/"$scene"/density_volume.npy \
    --config=configs/syn.json \
    --dataset_type=nerf \
    --test_hold_every=2 \
    --high_fq
done