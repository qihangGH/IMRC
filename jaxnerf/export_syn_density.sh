#!/bin/bash
DATA_ROOT=path/to/data/nerf_synthetic
TRAIN_ROOT=jaxnerf/tmp/blender
SCENES="chair lego drums ficus hotdog materials mic ship"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python -m jaxnerf.export_density \
    --train_dir="$TRAIN_ROOT"/"$scene" \
    --data_dir="$DATA_ROOT"/"$scene" \
    --config=jaxnerf/configs/blender \
    --reso=512,512,512 \
    --radius=1.5,1.5,1.5 \
    --export_with_fine=True \
    --export_chunk_size=524288
done
