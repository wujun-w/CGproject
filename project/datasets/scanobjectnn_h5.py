import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from project.config import SCANOBJECTNN_CLASS_NAMES
from project.datasets.transforms import build_points


def load_scanobjectnn_class_names(data_root):
    mapping_path = Path(data_root) / "label_mapping.csv"
    if not mapping_path.exists():
        return SCANOBJECTNN_CLASS_NAMES

    rows = list(csv.DictReader(mapping_path.open("r", encoding="utf-8")))
    rows.sort(key=lambda row: int(row["Label"]))
    return [row["Category"].strip() for row in rows]


class ScanObjectNNH5Dataset(Dataset):
    def __init__(self, data_root, split, num_points, training):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.training = training
        # 优先读取数据目录中的官方标签映射，避免手写顺序出错。
        self.class_names = load_scanobjectnn_class_names(self.data_root)

        # 固定使用课程项目选定的官方 split。
        split_to_file = {
            "train": self.data_root / "training_objectdataset.h5",
            "test": self.data_root / "test_objectdataset.h5",
        }

        with h5py.File(split_to_file[split], "r") as handle:
            self.points = handle["data"][:].astype(np.float32)
            self.labels = handle["label"][:].astype(np.int64).reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = build_points(self.points[index], num_points=self.num_points, training=self.training)
        label = int(self.labels[index])

        # ScanObjectNN 原始 H5 没有单独的文件名，这里生成一个稳定 ID 便于记录错误样本。
        meta = {
            "sample_index": index,
            "shape_id": f"scanobjectnn_{self.split}_{index:05d}",
            "shape_name": self.class_names[label],
            "split": self.split,
        }

        return torch.from_numpy(points), torch.tensor(label, dtype=torch.long), meta
