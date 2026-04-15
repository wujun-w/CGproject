import csv
import json
from pathlib import Path

import torch


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_experiment_dirs(results_root, exp_name):
    # 每组实验单独建目录，保证四组结果天然分开存放。
    root = ensure_dir(Path(results_root) / exp_name)
    return {
        "root": root,
        "logs": ensure_dir(root / "logs"),
        "metrics": ensure_dir(root / "metrics"),
        "plots": ensure_dir(root / "plots"),
        "checkpoints": ensure_dir(root / "checkpoints"),
        "exports": ensure_dir(root / "exports"),
    }


def sanitize_for_json(data):
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [sanitize_for_json(value) for value in data]
    if isinstance(data, Path):
        return str(data)
    return data


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(sanitize_for_json(payload), indent=2), encoding="utf-8")


def save_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # 结果 CSV 统一按首行字段写出，便于后续脚本再次读取。
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_model_checkpoint(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
