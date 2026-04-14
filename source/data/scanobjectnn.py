from pathlib import Path

from .common import BasePointCloudDataset, load_h5_file, read_class_names


class ScanObjectNNH5(BasePointCloudDataset):
    SPLIT_FILES = {
        "train": "training_objectdataset_augmentedrot_scale75.h5",
        "test": "test_objectdataset_augmentedrot_scale75.h5",
    }

    def __init__(
        self,
        root,
        split="train",
        num_points=1024,
        normalize=True,
        augment=False,
        augment_cfg=None,
        split_file=None,
    ):
        root = Path(root)
        file_name = split_file or self.SPLIT_FILES[split]
        file_path = root / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"ScanObjectNN file not found: {file_path}. "
                "If your files use different names, pass --scan-train-file or --scan-test-file."
            )

        points, labels = load_h5_file(file_path)
        self.class_names = read_class_names(root / "classes.txt")
        super().__init__(
            points=points,
            labels=labels,
            num_points=num_points,
            train=split == "train",
            normalize=normalize,
            augment=augment,
            augment_cfg=augment_cfg,
        )
