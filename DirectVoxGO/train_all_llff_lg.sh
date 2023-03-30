#!/bin/bash
SCENES="fern flower fortress horns leaves orchids room trex"
export CUDA_VISIBLE_DEVICES=0
# launch training jobs for all scenes.
for scene in $SCENES; do
   python run.py \
      --config=/lkq/fqh/dvgo/configs/llff/"$scene"_lg.py \
      --render_test
done
