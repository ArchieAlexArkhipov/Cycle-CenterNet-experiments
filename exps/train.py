import os
import os.path as osp
from sys import argv

import mmcv
import wandb
from mmcv import Config
from mmdet.apis import set_random_seed, train_detector

# Let's take a look at the dataset image
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import subprocess
import random


def get_free_gpu():
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


os.environ["WANDB_CACHE_DIR"] = "/home/aiarhipov/.cache/wandb"
os.environ["WANDB_CONFIG_DIR"] = "/home/aiarhipov/.config/wandb"
os.environ["WANDB_DIR"] = "/home/aiarhipov/centernet/exps/wandb"
wandb.login()
cfg = Config.fromfile(f"/home/aiarhipov/centernet/exps/{argv[1]}config.py")


set_random_seed(0, deterministic=False)
cfg.gpu_ids = [get_free_gpu()]
val = True
# Build dataset
if len(argv) == 2:
    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)]
elif argv[2] == "no-val":
    datasets = [build_dataset(cfg.data.train)]
    cfg.workflow = [("train", 1)]
    val = False

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=val)
