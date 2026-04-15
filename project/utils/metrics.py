import numpy as np


def compute_classification_metrics(preds, labels, class_names):
    # 课程项目只保留 OA、mAcc 和 per-class accuracy 三类核心指标。
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    oa = float((preds == labels).mean())
    per_class_acc = {}
    class_acc_values = []

    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if mask.sum() == 0:
            class_acc = 0.0
        else:
            class_acc = float((preds[mask] == labels[mask]).mean())
        per_class_acc[class_name] = class_acc
        class_acc_values.append(class_acc)

    macc = float(np.mean(class_acc_values)) if class_acc_values else 0.0
    return {
        "oa": oa,
        "macc": macc,
        "per_class_acc": per_class_acc,
    }
