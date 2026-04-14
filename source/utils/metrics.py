import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


def classification_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    overall_acc = float((y_true == y_pred).mean())
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_acc = []
    for class_idx in range(num_classes):
        class_total = confusion[class_idx].sum()
        acc = float(confusion[class_idx, class_idx] / class_total) if class_total > 0 else 0.0
        per_class_acc.append(acc)
    return {
        "overall_acc": overall_acc,
        "mean_class_acc": float(np.mean(per_class_acc)),
        "per_class_acc": per_class_acc,
        "confusion_matrix": confusion,
    }


def save_confusion_matrix_csv(matrix, class_names, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + list(class_names))
        for idx, row in enumerate(matrix):
            row_name = class_names[idx] if idx < len(class_names) else str(idx)
            writer.writerow([row_name] + row.tolist())


def save_per_class_accuracy_csv(per_class_acc, class_names, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "accuracy"])
        for idx, acc in enumerate(per_class_acc):
            name = class_names[idx] if idx < len(class_names) else str(idx)
            writer.writerow([name, f"{acc:.6f}"])
