import os.path as osp

import mmcv
import torch
import wandb
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmdet.apis import set_random_seed, train_detector

# Let's take a look at the dataset image
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

wandb.login()


cfg = Config.fromfile("/home/aiarhipov/MM/configs/5_resnet34_wandb.py")


set_random_seed(0, deterministic=False)

# Build dataset
datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)]

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
