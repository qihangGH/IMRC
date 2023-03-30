#!/bin/bash
#JaxNeRF
DATA_ROOT=path/to/data/nerf_synthetic
ROOT_DIR=../../jaxnerf/tmp/blender
SCENES="chair lego drums ficus hotdog materials mic ship"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
    python opt_metric_from_other_models.py \
      "$DATA_ROOT"/"$scene" \
      "$ROOT_DIR"/"$scene"/density_volume_fine_512.npy \
      --config=configs/syn.json \
      --basis_dim=9 \
      --dataset_type=nerf \
      --test_hold_every=2 \
      --high_fq
done
