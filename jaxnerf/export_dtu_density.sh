#!/bin/bash
DATA_ROOT=path/to/public_data
SCENES="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
export CUDA_VISIBLE_DEVICES=0
for scene in $SCENES; do
  python -m jaxnerf.export_density \
    --train_dir=jaxnerf/tmp/dtu_"$scene" \
    --data_dir="$DATA_ROOT"/dtu_scan"$scene" \
    --config=jaxnerf/configs/dtu \
    --reso=512,512,512 \
    --radius=1.0,1.0,1.0 \
    --export_with_fine=True \
    --export_chunk_size=524288
done