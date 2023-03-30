#!/bin/bash
#JaxNeRF
DATA_ROOT=path/to/data/nerf_llff_data
ROOT_DIR=../../jaxnerf/jaxnerf_models/llff
SCENES="fern flower fortress horns leaves orchids room trex"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python opt_metric_from_other_models.py \
    "$DATA_ROOT"/"$scene" \
    "$ROOT_DIR"/"$scene"/density_volume_fine_598.npy \
    --config=configs/llff_color_metric.json \
    --use_ndc \
    --dataset_type=llff \
    --high_fq
done
