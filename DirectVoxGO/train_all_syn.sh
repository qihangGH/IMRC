#!/bin/bash
SCENES="lego chair drums ficus hotdog materials mic ship"
# launch training jobs for all scenes.
for scene in $SCENES; do
    CUDA_VISIBLE_DEVICES=5 python run.py \
    --config=/lkq/fqh/dvgo/configs/nerf/"$scene".py \
    --render_test
done
