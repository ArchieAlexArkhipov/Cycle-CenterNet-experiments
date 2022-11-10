import os.path as osp

import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed, train_detector

# Let's take a look at the dataset image
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

# cfg = Config.fromfile("/home/aiarhipov/MM/config.py")


# # Modify dataset type and path
# cfg.dataset_type = "CocoDataset"
# classes = ("box",)
# cfg.data_root = "/home/aiarhipov/datasets/WTW-dataset/"

# cfg.data.test.type = "CocoDataset"
# cfg.data.test.data_root = "/home/aiarhipov/datasets/WTW-dataset/"
# cfg.data.test.ann_file = "/home/aiarhipov/datasets/WTW-dataset/test/test.json"
# cfg.data.test.img_prefix = "test/images/"
# cfg.data.test.classes = classes

# cfg.data.train.type = "CocoDataset"
# cfg.data.train.data_root = "/home/aiarhipov/datasets/WTW-dataset/"
# cfg.data.train.ann_file = (
#     "/home/aiarhipov/datasets/WTW-dataset/train/train.json"
# )
# cfg.data.train.img_prefix = "train/images/"
# cfg.data.train.classes = classes

# cfg.data.val.type = "CocoDataset"
# cfg.data.val.data_root = "/home/aiarhipov/datasets/WTW-dataset/"
# cfg.data.val.ann_file = "/home/aiarhipov/datasets/WTW-dataset/test/test.json"
# cfg.data.val.img_prefix = "test/images/"
# cfg.data.val.classes = classes

# # modify num classes of the model in box head
# cfg.model.bbox_head.num_classes = 1
# # If we need to finetune a model based on a pre-trained detector, we need to
# # use load_from to set the path of checkpoints.
# cfg.load_from = "/home/aiarhipov/MM/tutorial_exps/16ep.pth"
# # cfg.load_from = 'checkpoints/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'

# # Set up working dir to save files and logs.
# cfg.work_dir = "./tutorial_exps"

# # The original learning rate (LR) is set for 8-GPU training.
# # We divide it by 8 since we only use one GPU.
# cfg.optimizer.lr = 0.02 / 8
# cfg.lr_config.warmup = None
# cfg.log_config.interval = 100

# # Change the evaluation metric since we use customized dataset.
# cfg.evaluation.metric = "bbox"  #'mAP'
# # We can set the evaluation interval to reduce the evaluation times
# cfg.evaluation.interval = 15
# # We can set the checkpoint saving interval to reduce the storage cost
# cfg.checkpoint_config.interval = 15

# # Set seed thus the results are more reproducible
# cfg.seed = 0
# set_random_seed(0, deterministic=False)
# cfg.gpu_ids = [6]
# cfg.device = "cuda"

# # We can also use tensorboard to log the training process
# cfg.log_config.hooks = [
#     dict(type="TextLoggerHook"),
#     dict(type="TensorboardLoggerHook"),
# ]

# cfg.runner.max_epochs = 56

# We can initialize the logger for training and have a look
# at the final config used for training
# print(f"Config:\n{cfg.pretty_text}")

cfg = Config.fromfile("/home/aiarhipov/MM/config_wtw_new.py")

set_random_seed(0, deterministic=False)

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
