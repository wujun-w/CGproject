import time

import numpy as np
import torch
from tqdm import tqdm

from project.utils.export import export_wrong_case_points
from project.utils.metrics import compute_classification_metrics


def evaluate_model(
    model,
    dataloader,
    device,
    class_names,
    desc,
    export_dir=None,
    experiment_name="",
    dataset_name="",
    model_name="",
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    wrong_cases = []

    start_time = time.time()

    with torch.no_grad():
        progress = tqdm(dataloader, desc=desc, leave=False, ncols=120)
        for points, labels, meta in progress:
            points = points.to(device)
            labels = labels.to(device)

            output = model(points, labels=labels)
            loss = output["loss"]
            preds = output["preds"]

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            points_np = points.detach().cpu().numpy()

            all_preds.append(preds_np)
            all_labels.append(labels_np)

            batch_oa = float((preds_np == labels_np).mean())
            avg_loss = total_loss / max(total_samples, 1)
            progress.set_postfix({"loss": f"{avg_loss:.4f}", "oa": f"{batch_oa:.4f}"})

            if export_dir is not None:
                # 最终测试时才导出错误样本，避免训练阶段产生大量无用文件。
                for index_in_batch, (pred_label, true_label) in enumerate(zip(preds_np, labels_np)):
                    if int(pred_label) == int(true_label):
                        continue

                    sample_index = int(meta["sample_index"][index_in_batch])
                    shape_id = str(meta["shape_id"][index_in_batch])
                    shape_name = str(meta["shape_name"][index_in_batch])

                    row = {
                        "experiment": experiment_name,
                        "dataset": dataset_name,
                        "model": model_name,
                        "split": "test",
                        "sample_index": sample_index,
                        "shape_id": shape_id,
                        "shape_name": shape_name,
                        "true_label": int(true_label),
                        "true_name": class_names[int(true_label)],
                        "pred_label": int(pred_label),
                        "pred_name": class_names[int(pred_label)],
                    }
                    wrong_cases.append(row)
                    export_wrong_case_points(
                        points=points_np[index_in_batch],
                        export_dir=export_dir,
                        row=row,
                    )

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_classification_metrics(
        preds=all_preds,
        labels=all_labels,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["eval_time_sec"] = time.time() - start_time
    metrics["num_samples"] = int(total_samples)
    metrics["wrong_case_count"] = len(wrong_cases)
    return metrics, wrong_cases
