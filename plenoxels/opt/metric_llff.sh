#!/bin/bash
DATA_ROOT=path/to/data/nerf_llff_data
ROOT_DIR=ckpt/llff_c2f_fasttv_10e
SCENES="fern flower fortress horns leaves orchids room trex"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python opt_metric_from_other_models.py \
    "$DATA_ROOT"/"$scene" \
    "$ROOT_DIR"/"$scene"/density_volume_598.npy \
    --config=configs/llff_color_metric.json \
    --dataset_type=llff \
    --use_ndc \
    --high_fq
done
