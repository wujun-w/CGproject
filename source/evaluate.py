import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ModelNet40H5, ScanObjectNNH5
from data.common import apply_vote_augmentation
from models import DGCNNCls, PointNetCls
from utils.io import ensure_dir, load_checkpoint, save_json
from utils.metrics import classification_metrics, save_confusion_matrix_csv, save_per_class_accuracy_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a point cloud classifier.")
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
    parser.add_argument("--vote-aug", action="store_true")
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


def apply_vote_batch(points, vote_aug):
    if not vote_aug:
        return points
    points_np = points.cpu().numpy()
    augmented = np.stack([apply_vote_augmentation(sample) for sample in points_np], axis=0)
    return torch.from_numpy(augmented).float()


@torch.no_grad()
def main():
    args = parse_args()
    args = align_args_with_checkpoint(args)
    dataset = build_dataset(args)
    max_label = int(dataset.labels.max())
    class_names = dataset.class_names or [str(i) for i in range(max_label + 1)]
    num_classes = max(len(class_names), max_label + 1)

    device = torch.device(args.device)
    model = build_model(args, num_classes).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    y_true, y_pred = [], []
    for points, labels in loader:
        logits_sum = None
        labels = labels.to(device)
        for vote_idx in range(max(args.vote_num, 1)):
            vote_points = apply_vote_batch(points, vote_aug=args.vote_aug and args.vote_num > 1 and vote_idx > 0)
            vote_points = vote_points.to(device).transpose(2, 1)
            logits, _ = model(vote_points)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        preds = logits_sum.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, num_classes)
    eval_dir = ensure_dir(Path(args.eval_root) / args.experiment)
    save_json(
        {"overall_acc": metrics["overall_acc"], "mean_class_acc": metrics["mean_class_acc"], "num_samples": len(y_true)},
        eval_dir / "metrics.json",
    )
    save_confusion_matrix_csv(metrics["confusion_matrix"], class_names, eval_dir / "confusion_matrix.csv")
    save_per_class_accuracy_csv(metrics["per_class_acc"], class_names, eval_dir / "per_class_accuracy.csv")

    with (eval_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_idx", "true_label", "pred_label"])
        for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            writer.writerow([idx, true_label, pred_label])

    print(f"overall_acc={metrics['overall_acc']:.4f}, mean_class_acc={metrics['mean_class_acc']:.4f}")


if __name__ == "__main__":
    main()
