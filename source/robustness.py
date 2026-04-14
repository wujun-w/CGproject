import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data import ModelNet40H5, ScanObjectNNH5
from data.common import apply_vote_augmentation, local_occlusion
from models import DGCNNCls, PointNetCls
from utils.io import ensure_dir, load_checkpoint, save_json
from utils.metrics import classification_metrics


def dropout_corruption(points, severity):
    ratio = min(0.2 * severity, 0.9)
    keep = max(1, int(points.shape[0] * (1.0 - ratio)))
    idx = np.random.choice(points.shape[0], keep, replace=False)
    sampled = points[idx]
    if sampled.shape[0] < points.shape[0]:
        pad_idx = np.random.choice(sampled.shape[0], points.shape[0] - sampled.shape[0], replace=True)
        sampled = np.concatenate([sampled, sampled[pad_idx]], axis=0)
    return sampled


def noise_corruption(points, severity):
    sigma = 0.01 * severity
    return points + np.random.normal(0.0, sigma, size=points.shape).astype(np.float32)


def rotation_corruption(points, severity):
    theta = (np.pi / 12.0) * severity
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation = np.array(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return points @ rotation.T


def crop_corruption(points, severity):
    severity_to_ratio = {1: 0.15, 2: 0.30, 3: 0.45}
    return local_occlusion(points, ratio=severity_to_ratio[severity], min_points=32)


def axis_crop_corruption(points, severity):
    threshold = -0.2 + 0.1 * severity
    mask = points[:, 0] > threshold
    cropped = points[mask]
    if len(cropped) < 8:
        return points
    idx = np.random.choice(len(cropped), points.shape[0], replace=len(cropped) < points.shape[0])
    return cropped[idx]


CORRUPTIONS = {
    "dropout": dropout_corruption,
    "noise": noise_corruption,
    "rotation": rotation_corruption,
    "crop": crop_corruption,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Robustness evaluation for point cloud classifiers.")
    parser.add_argument("--model", choices=["pointnet", "dgcnn"], required=True)
    parser.add_argument("--dataset", choices=["modelnet40", "scanobjectnn"], required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--scan-test-file", type=str, default=None)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feature-transform", action="store_true")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--emb-dims", type=int, default=1024)
    parser.add_argument("--vote-num", type=int, default=1)
    parser.add_argument("--crop-mode", choices=["local_occlusion", "axis_cut"], default="local_occlusion")
    parser.add_argument("--eval-root", type=str, default="results/eval_logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def align_args_with_checkpoint(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    args.dropout = float(saved_args.get("dropout", args.dropout))
    args.feature_transform = bool(saved_args.get("feature_transform", args.feature_transform))
    args.k = int(saved_args.get("k", args.k))
    args.emb_dims = int(saved_args.get("emb_dims", args.emb_dims))
    return args


def build_dataset(args):
    if args.dataset == "modelnet40":
        return ModelNet40H5(args.data_root, split="test", num_points=args.num_points, augment=False)
    return ScanObjectNNH5(
        args.data_root,
        split="test",
        num_points=args.num_points,
        augment=False,
        split_file=args.scan_test_file,
    )


def build_model(args, num_classes):
    if args.model == "pointnet":
        return PointNetCls(num_classes=num_classes, feature_transform=args.feature_transform, dropout=args.dropout)
    return DGCNNCls(num_classes=num_classes, k=args.k, emb_dims=args.emb_dims, dropout=args.dropout)


def apply_vote_batch(points, enable_vote_aug):
    if not enable_vote_aug:
        return points
    points_np = points.cpu().numpy()
    augmented = np.stack([apply_vote_augmentation(sample) for sample in points_np], axis=0)
    return torch.from_numpy(augmented).float()


class CorruptedDataset(Dataset):
    def __init__(self, base_dataset, corruption_name, severity, crop_mode="local_occlusion"):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.crop_mode = crop_mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        points, label = self.base_dataset[index]
        points_np = points.numpy().copy()
        if self.corruption_name == "crop" and self.crop_mode == "axis_cut":
            points_np = axis_crop_corruption(points_np, self.severity).astype(np.float32)
        else:
            points_np = CORRUPTIONS[self.corruption_name](points_np, self.severity).astype(np.float32)
        return torch.from_numpy(points_np), label


@torch.no_grad()
def evaluate(model, loader, device, num_classes, vote_num=1):
    model.eval()
    y_true, y_pred = [], []
    for points, labels in loader:
        logits_sum = None
        labels = labels.to(device)
        for vote_idx in range(max(vote_num, 1)):
            vote_points = apply_vote_batch(points, enable_vote_aug=vote_num > 1 and vote_idx > 0)
            vote_points = vote_points.to(device).transpose(2, 1)
            logits, _ = model(vote_points)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        preds = logits_sum.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return classification_metrics(y_true, y_pred, num_classes)


def main():
    args = parse_args()
    args = align_args_with_checkpoint(args)
    base_dataset = build_dataset(args)
    max_label = int(base_dataset.labels.max())
    class_names = base_dataset.class_names or [str(i) for i in range(max_label + 1)]
    num_classes = max(len(class_names), max_label + 1)

    device = torch.device(args.device)
    model = build_model(args, num_classes).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    out_dir = ensure_dir(Path(args.eval_root) / args.experiment)
    summary = {}
    for corruption_name in CORRUPTIONS:
        summary[corruption_name] = {}
        for severity in [1, 2, 3]:
            dataset = CorruptedDataset(base_dataset, corruption_name, severity, crop_mode=args.crop_mode)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            metrics = evaluate(model, loader, device, num_classes, vote_num=args.vote_num)
            summary[corruption_name][f"severity_{severity}"] = {
                "overall_acc": metrics["overall_acc"],
                "mean_class_acc": metrics["mean_class_acc"],
            }
            print(
                f"{corruption_name}/severity_{severity}: "
                f"overall_acc={metrics['overall_acc']:.4f}, "
                f"mean_class_acc={metrics['mean_class_acc']:.4f}"
            )

    save_json(summary, out_dir / "robustness_summary.json")


if __name__ == "__main__":
    main()
