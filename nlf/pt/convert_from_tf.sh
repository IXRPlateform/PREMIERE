#!/usr/bin/env bash

set -euo pipefail

python -m nlf.pt.convert_ckpt_from_tf --input-model-path=models/tf/nlf_l_crop_tf --output-model-path=models/pt/nlf_l_crop.pt --config-name=nlf_l
python -m nlf.pt.convert_ckpt_from_tf --input-model-path=models/tf/nlf_s_crop_tf --output-model-path=models/pt/nlf_s_crop.pt --config-name=nlf_s

python -m nlf.pt.multiperson.multiperson_model --input-model-path=models/pt/nlf_l_crop.pt --output-model-path=models/pt/nlf_l_multi.torchscript --config-name=nlf_l
python -m nlf.pt.multiperson.multiperson_model --input-model-path=models/pt/nlf_s_crop.pt --output-model-path=models/pt/nlf_s_multi.torchscript --config-name=nlf_s