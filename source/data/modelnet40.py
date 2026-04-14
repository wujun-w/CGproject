from pathlib import Path

import numpy as np

from .common import BasePointCloudDataset, load_h5_file, read_class_names


class ModelNet40H5(BasePointCloudDataset):
    def __init__(self, root, split="train", num_points=1024, normalize=True, augment=False, augment_cfg=None):
        root = Path(root)
        pattern = "ply_data_train*.h5" if split == "train" else "ply_data_test*.h5"
        files = sorted(root.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No ModelNet40 H5 files found under {root}. Expected pattern: {pattern}"
            )

        all_points = []
        all_labels = []
        for file_path in files:
            points, labels = load_h5_file(file_path)
            all_points.append(points)
            all_labels.append(labels)

        self.class_names = read_class_names(root / "shape_names.txt")
        points = np.concatenate(all_points, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        super().__init__(
            points=points,
            labels=labels,
            num_points=num_points,
            train=split == "train",
            normalize=normalize,
            augment=augment,
            augment_cfg=augment_cfg,
        )
