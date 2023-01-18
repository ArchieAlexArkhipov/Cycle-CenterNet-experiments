#!/bin/bash

conda activate mmdet_edit
python /home/aiarhipov/mmdetection/tools/test.py \
    /home/aiarhipov/centernet/exps/16_paper_params_dla34_batch8/config.py \
    /home/aiarhipov/centernet/exps/16_paper_params_dla34_batch8/epoch_150.pth \
    --out /home/aiarhipov/centernet/exps/16_paper_params_dla34_batch8/res_output.pkl \
    --eval bbox
python /home/aiarhipov/mmdetection/tools/analysis_tools/analyze_results.py \
    /home/aiarhipov/centernet/exps/15_paper_params_resnet34_batch8/config.py \
    /home/aiarhipov/centernet/exps/15_paper_params_resnet34_batch8/res_output.pkl \
    /home/aiarhipov/centernet/exps/15_paper_params_resnet34_batch8/ \
    --topk 50 \
    --show-score-thr 0.3