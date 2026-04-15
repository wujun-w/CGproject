import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from project.datasets.transforms import build_points


class ModelNet40H5Dataset(Dataset):
    def __init__(self, data_root, split, num_points, training):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.training = training

        # 训练阶段直接读取预处理后的 H5，避免每次从 OFF 网格重新采样。
        metadata_path = self.data_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.class_names = metadata["class_names"]

        file_name = metadata["files"][split]
        with h5py.File(self.data_root / file_name, "r") as handle:
            self.points = handle["data"][:].astype(np.float32)
            self.labels = handle["label"][:].astype(np.int64).reshape(-1)
            self.shape_ids = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle["shape_id"][:]]
            self.shape_names = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle["shape_name"][:]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = build_points(self.points[index], num_points=self.num_points, training=self.training)
        label = int(self.labels[index])

        # meta 仅保留课程报告和错误样本分析所需的最小字段。
        meta = {
            "sample_index": index,
            "shape_id": self.shape_ids[index],
            "shape_name": self.shape_names[index],
            "split": self.split,
        }

        return torch.from_numpy(points), torch.tensor(label, dtype=torch.long), meta
