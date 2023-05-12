#!/bin/bash
conda init bash
conda activate mmdet_edit
python /home/aiarhipov/mmdetection/tools/test.py \
    /home/aiarhipov/centernet/exps/20_cycle_l1/config.py \
    /home/aiarhipov/centernet/exps/20_cycle_middle_pairing_loss_speedup/epoch_30.pth \
    --out /home/aiarhipov/centernet/exps/20_cycle_l1/res_output.pkl \
    --eval bbox \
    --gpu-id 2
# python /home/aiarhipov/mmdetection/tools/analysis_tools/analyze_results.py \
#     /home/aiarhipov/centernet/exps/20_cycle_l1/config.py \
#     /home/aiarhipov/centernet/exps/20_cycle_l1/res_output.pkl \
#     /home/aiarhipov/centernet/exps/20_cycle_l1/ \
#     --topk 30 \
#     --show-score-thr 0.7