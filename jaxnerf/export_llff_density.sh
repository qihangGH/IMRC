#!/bin/bash
DATA_ROOT=path/to/data/nerf_llff_data
SCENES="fern flower fortress horns leaves orchids room trex"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python -m jaxnerf.export_density \
    --train_dir=jaxnerf/tmp/llff/"$scene" \
    --data_dir="$DATA_ROOT"/"$scene" \
    --config=jaxnerf/configs/llff \
    --reso=598,665,400 \
    --radius=1.496031746031746,1.6613756613756614,1.0 \
    --export_with_fine=True \
    --export_chunk_size=524288
done
