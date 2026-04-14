import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ModelNet40H5, ScanObjectNNH5
from models import DGCNNCls, PointNetCls
from models.pointnet import feature_transform_regularizer
from utils.io import append_csv_row, ensure_dir, save_checkpoint, save_json
from utils.metrics import classification_metrics, save_confusion_matrix_csv, save_per_class_accuracy_csv
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a point cloud classifier.")
    parser.add_argument("--model", choices=["pointnet", "dgcnn"], required=True)
    parser.add_argument("--dataset", choices=["modelnet40", "scanobjectnn"], required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--scan-train-file", type=str, default=None)
    parser.add_argument("--scan-test-file", type=str, default=None)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--emb-dims", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", type=str, default="results/train_logs")
    parser.add_argument("--eval-root", type=str, default="results/eval_logs")
    parser.add_argument("--ckpt-root", type=str, default="checkpoints")
    parser.add_argument("--feature-transform", action="store_true")
    parser.add_argument("--occlusion-prob", type=float, default=0.4)
    parser.add_argument("--occlusion-min-ratio", type=float, default=0.1)
    parser.add_argument("--occlusion-max-ratio", type=float, default=0.2)
    parser.add_argument("--scale-low", type=float, default=0.9)
    parser.add_argument("--scale-high", type=float, default=1.1)
    parser.add_argument("--rotation-perturb-deg", type=float, default=10.0)
    parser.add_argument("--disable-legacy-dropout", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_augment_cfg(args):
    return {
        "use_dropout": not args.disable_legacy_dropout,
        "dropout_max_ratio": 0.2,
        "scale_low": args.scale_low,
        "scale_high": args.scale_high,
        "rotation_perturb_deg": args.rotation_perturb_deg,
        "z_rotate_max_deg": 30.0,
        "translate_range": 0.1,
        "jitter_sigma": 0.01,
        "jitter_clip": 0.05,
        "occlusion_prob": args.occlusion_prob,
        "occlusion_min_ratio": args.occlusion_min_ratio,
        "occlusion_max_ratio": args.occlusion_max_ratio,
        "occlusion_min_points": 32,
    }


def build_datasets(args):
    augment_cfg = build_augment_cfg(args)
    if args.dataset == "modelnet40":
        train_set = ModelNet40H5(
            args.data_root,
            split="train",
            num_points=args.num_points,
            augment=True,
            augment_cfg=augment_cfg,
        )
        test_set = ModelNet40H5(args.data_root, split="test", num_points=args.num_points, augment=False)
    else:
        train_set = ScanObjectNNH5(
            args.data_root,
            split="train",
            num_points=args.num_points,
            augment=True,
            augment_cfg=augment_cfg,
            split_file=args.scan_train_file,
        )
        test_set = ScanObjectNNH5(
            args.data_root,
            split="test",
            num_points=args.num_points,
            augment=False,
            split_file=args.scan_test_file,
        )
    return train_set, test_set


def build_model(args, num_classes):
    if args.model == "pointnet":
        return PointNetCls(num_classes=num_classes, feature_transform=args.feature_transform, dropout=args.dropout)
    return DGCNNCls(num_classes=num_classes, k=args.k, emb_dims=args.emb_dims, dropout=args.dropout)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true, y_pred = [], []
    for points, labels in loader:
        points = points.to(device).transpose(2, 1)
        labels = labels.to(device)
        logits, _ = model(points)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def train_one_epoch(model, loader, optimizer, device, args, num_classes):
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true, y_pred = [], []

    progress = tqdm(loader, leave=False)
    for points, labels in progress:
        points = points.to(device).transpose(2, 1)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, trans_feat = model(points)
        loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
        if args.model == "pointnet" and args.feature_transform:
            loss = loss + 0.001 * feature_transform_regularizer(trans_feat)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        progress.set_description(f"loss={loss.item():.4f}")

    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    train_set, test_set = build_datasets(args)
    max_label = int(max(train_set.labels.max(), test_set.labels.max()))
    class_names = train_set.class_names or [str(i) for i in range(max_label + 1)]
    num_classes = max(len(class_names), max_label + 1)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = ensure_dir(Path(args.ckpt_root) / args.experiment)
    train_log_dir = ensure_dir(Path(args.log_root) / args.experiment)
    eval_log_dir = ensure_dir(Path(args.eval_root) / args.experiment)
    save_json(vars(args), train_log_dir / "config.json")

    best_acc = -1.0
    start_time = time.time()
    epoch_csv = train_log_dir / "epoch_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, num_classes)
        test_metrics = evaluate(model, test_loader, device, num_classes)
        scheduler.step()

        append_csv_row(
            epoch_csv,
            ["epoch", "train_loss", "train_acc", "train_mean_class_acc", "val_loss", "val_acc", "val_mean_class_acc", "lr"],
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_acc": f"{train_metrics['overall_acc']:.6f}",
                "train_mean_class_acc": f"{train_metrics['mean_class_acc']:.6f}",
                "val_loss": f"{test_metrics['loss']:.6f}",
                "val_acc": f"{test_metrics['overall_acc']:.6f}",
                "val_mean_class_acc": f"{test_metrics['mean_class_acc']:.6f}",
                "lr": f"{scheduler.get_last_lr()[0]:.8f}",
            },
        )

        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, epoch, best_acc, args)
        if test_metrics["overall_acc"] > best_acc:
            best_acc = test_metrics["overall_acc"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_acc, args)
            save_confusion_matrix_csv(test_metrics["confusion_matrix"], class_names, eval_log_dir / "best_confusion_matrix.csv")
            save_per_class_accuracy_csv(test_metrics["per_class_acc"], class_names, eval_log_dir / "best_per_class_accuracy.csv")
            save_json(
                {
                    "best_epoch": epoch,
                    "best_overall_acc": test_metrics["overall_acc"],
                    "best_mean_class_acc": test_metrics["mean_class_acc"],
                    "elapsed_minutes": (time.time() - start_time) / 60.0,
                },
                eval_log_dir / "best_metrics.json",
            )

        print(
            f"[Epoch {epoch:03d}] train_acc={train_metrics['overall_acc']:.4f} "
            f"val_acc={test_metrics['overall_acc']:.4f} best_acc={best_acc:.4f}"
        )

    save_json(
        {
            "total_minutes": (time.time() - start_time) / 60.0,
            "best_acc": best_acc,
            "experiment": args.experiment,
        },
        train_log_dir / "run_summary.json",
    )


if __name__ == "__main__":
    main()
