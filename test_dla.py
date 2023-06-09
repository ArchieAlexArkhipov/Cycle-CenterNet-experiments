# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import (
    build_ddp,
    build_dp,
    compat_cfg,
    get_device,
    replace_cfg_vals,
    setup_multi_processes,
    update_data_root,
)
import subprocess
import random


def get_free_gpu(force_gpu=None):
    if force_gpu:
        return force_gpu
    else:
        log = str(
            subprocess.check_output("nvidia-smi --format=csv --query-gpu=utilization.gpu,memory.used", shell=True)
        ).split(r"\n")[1:-1]
        free_gpu = []
        for idx, gpu_info in enumerate(log):
            if gpu_info[:-4].split(" %, ")[0] == "0" and gpu_info[:-4].split(" %, ")[1] == "3":
                free_gpu.append(idx)
        if free_gpu:
            return random.choice(free_gpu)
        raise RuntimeError("All gpus are used")


CONFIG = "/home/aiarhipov/centernet/exps/27_seg/config.py"
CHECKPOINT = "/home/aiarhipov/centernet/exps/27_seg/latest.pth"
OUTPUT_PATH = "/home/aiarhipov/centernet/exps/27_seg/test_output.pkl"
EXP_DIR = "/home/aiarhipov/centernet/exps/27_seg"
cfg = Config.fromfile(CONFIG)

# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)

# update data root according to MMDET_DATASETS
update_data_root(cfg)

cfg = compat_cfg(cfg)

# set multi-process settings
setup_multi_processes(cfg)

# set cudnn_benchmark
if cfg.get("cudnn_benchmark", False):
    torch.backends.cudnn.benchmark = True

if "pretrained" in cfg.model:
    cfg.model.pretrained = None
elif "init_cfg" in cfg.model.backbone:
    cfg.model.backbone.init_cfg = None

if cfg.model.get("neck"):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get("rfp_backbone"):
                if neck_cfg.rfp_backbone.get("pretrained"):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get("rfp_backbone"):
        if cfg.model.neck.rfp_backbone.get("pretrained"):
            cfg.model.neck.rfp_backbone.pretrained = None


cfg.gpu_ids = [get_free_gpu()]

cfg.device = get_device()
# init distributed env first, since logger depends on the dist info.

distributed = False


test_dataloader_default_args = dict(samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

# in case the test dataset is concatenated
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

test_loader_cfg = {**test_dataloader_default_args, **cfg.data.get("test_dataloader", {})}

rank, _ = get_dist_info()
# allows not to create

# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, **test_loader_cfg)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
fp16_cfg = cfg.get("fp16", None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, CHECKPOINT, map_location="cpu")

# old versions did not save class info in checkpoints, this walkaround is
# for backward compatibility
if "CLASSES" in checkpoint.get("meta", {}):
    model.CLASSES = checkpoint["meta"]["CLASSES"]
else:
    model.CLASSES = dataset.CLASSES

if not distributed:
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader, False, EXP_DIR, 0.3)
else:
    model = build_ddp(model, cfg.device, device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False)
    outputs = multi_gpu_test(
        model,
        data_loader,
        EXP_DIR,
        True or cfg.evaluation.get("gpu_collect", False),
    )

rank, _ = get_dist_info()
if rank == 0:
    if True:
        print(f"\nwriting results to {OUTPUT_PATH}")
        mmcv.dump(outputs, OUTPUT_PATH)
    kwargs = {}

    if True:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        # hard-code way to remove EvalHook args
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric="bbox", **kwargs))
        metric = dataset.evaluate(outputs, **eval_kwargs)
        print(metric)
        metric_dict = dict(config=cfg, metric=metric)
        if args.work_dir is not None and rank == 0:
            mmcv.dump(metric_dict, json_file)
