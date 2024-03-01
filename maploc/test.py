
import os.path as osp
from typing import Optional
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from . import logger, pl_logger, EXPERIMENTS_PATH
from .data import modules as data_modules
from .module import GenericModule
from .data.boreas.dataset import BoreasDataModule
from .data.kitti.dataset import KittiDataModule
from .data.radiate.dataset import RadiateDataModule
cfg = {
    "name": "radiate",
    # paths and fetch
    "data_dir": "/home/classlab2/root/OrienterNet/datasets/kitti",
    "tiles_filename": "tiles.pkl",
    "splits": {
        "train": "test3_files.txt",
        "val": "test3_files.txt",
        "test": "test3_files.txt",
    },
    "loading": {
        "train": {"batch_size": 4, "num_workers": 4},
        "val": {"batch_size": 4, "num_workers": 4},
        "test": {"batch_size": 1, "num_workers": 0},
    },
    "max_num_val": 500,
    "selection_subset_val": "furthest",
    "drop_train_too_close_to_val": 5.0,
    "skip_frames": 1,
    "camera_index": 2,
    # overwrite
    "crop_size_meters": 64,
    "max_init_error": 20,
    "max_init_error_rotation": 10,
    "add_map_mask": True,
    "mask_pad": 2,
    "target_focal_length": 256,
}

# 实例化数据模块
data_module = KittiDataModule(cfg)
data_module.setup()
# 获取训练集数据加载器
train_dataloader = data_module.dataloader()
print("Number of training batches:", len(train_dataloader))
for batch in train_dataloader:
    # 在这里添加你的自定义逻辑来检查数据
    # 例如，你可以打印一些张量的形状等
    print("Batch shape:", {key: value.shape for key, value in batch.items()})
    break
