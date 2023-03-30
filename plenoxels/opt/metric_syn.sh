#!/bin/bash
DATA_ROOT=path/to/data/nerf_synthetic
ROOT_DIR=ckpt/nerf_synthetic
SCENES="chair lego drums ficus hotdog materials mic ship"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python opt_metric.py \
    "$DATA_ROOT"/"$scene" \
    "$ROOT_DIR"/"$scene"/ckpt.npz \
    --config=configs/syn.json \
    --dataset_type=nerf \
    --test_hold_every=2 \
    --high_fq
done