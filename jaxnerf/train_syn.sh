#!/bin/bash
DATA_ROOT=path/to/data
ROOT_DIR=path/to/jaxnerf/tmp/blender
SCENES="lego chair drums ficus hotdog materials mic ship"
DATA_FOLDER="nerf_synthetic"
export CUDA_VISIBLE_DEVICES=0
# launch training jobs for all scenes.
for scene in $SCENES; do
  python -m jaxnerf.train \
    --data_dir="$DATA_ROOT"/"$DATA_FOLDER"/"$scene" \
    --train_dir="$ROOT_DIR"/"$scene" \
    --config=jaxnerf/configs/blender
done
