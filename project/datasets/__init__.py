from pathlib import Path

from project.datasets.modelnet40_h5 import ModelNet40H5Dataset
from project.datasets.scanobjectnn_h5 import ScanObjectNNH5Dataset


def build_dataset(dataset_name, data_root, split, num_points, training):
    data_root = Path(data_root)

    if dataset_name == "modelnet40_h5":
        return ModelNet40H5Dataset(
            data_root=data_root,
            split=split,
            num_points=num_points,
            training=training,
        )

    if dataset_name == "scanobjectnn_h5":
        return ScanObjectNNH5Dataset(
            data_root=data_root,
            split=split,
            num_points=num_points,
            training=training,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")
