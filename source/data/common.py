from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points


def random_sample_points(points, num_points):
    if points.shape[0] == num_points:
        return points
    replace = points.shape[0] < num_points
    idx = np.random.choice(points.shape[0], num_points, replace=replace)
    return points[idx]


def uniform_resample_points(points, num_points):
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if points.shape[0] == num_points:
        return points
    replace = points.shape[0] < num_points
    idx = np.random.choice(points.shape[0], num_points, replace=replace)
    return points[idx]


def jitter_points(points, sigma=0.01, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + noise


def random_point_dropout(points, max_dropout_ratio=0.2):
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random(points.shape[0]) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        points[drop_idx, :] = points[0, :]
    return points


def random_translate(points, shift_range=0.1):
    return points + np.random.uniform(-shift_range, shift_range, 3)


def random_scale(points, scale_low=0.9, scale_high=1.1):
    scale = np.random.uniform(scale_low, scale_high)
    return points * scale


def random_rotate_z(points, max_degrees=30.0):
    max_radians = np.deg2rad(max_degrees)
    theta = np.random.uniform(-max_radians, max_radians)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation = np.array(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return points @ rotation.T


def random_rotation_perturb(points, max_degrees=10.0):
    max_radians = np.deg2rad(max_degrees)
    angles = np.random.uniform(-max_radians, max_radians, 3)
    rx, ry, rz = angles
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rotation = rot_z @ rot_y @ rot_x
    return points @ rotation.T


def local_occlusion(points, ratio, min_points=32):
    num_points = points.shape[0]
    if num_points <= min_points:
        return points
    num_remove = int(num_points * ratio)
    num_remove = min(max(num_remove, 1), num_points - min_points)
    if num_remove <= 0:
        return points

    center_idx = np.random.randint(0, num_points)
    center = points[center_idx]
    distances = np.linalg.norm(points - center, axis=1)
    keep_idx = np.argsort(distances)[num_remove:]
    kept = points[keep_idx]
    return uniform_resample_points(kept, num_points)


def apply_training_augmentation(points, cfg):
    if cfg.get("use_dropout", True):
        points = random_point_dropout(points, max_dropout_ratio=cfg.get("dropout_max_ratio", 0.2))
    points = random_scale(points, scale_low=cfg.get("scale_low", 0.9), scale_high=cfg.get("scale_high", 1.1))
    points = random_rotation_perturb(points, max_degrees=cfg.get("rotation_perturb_deg", 10.0))
    points = random_rotate_z(points, max_degrees=cfg.get("z_rotate_max_deg", 30.0))
    points = random_translate(points, shift_range=cfg.get("translate_range", 0.1))
    points = jitter_points(points, sigma=cfg.get("jitter_sigma", 0.01), clip=cfg.get("jitter_clip", 0.05))
    if np.random.random() < cfg.get("occlusion_prob", 0.0):
        ratio = np.random.uniform(cfg.get("occlusion_min_ratio", 0.1), cfg.get("occlusion_max_ratio", 0.2))
        points = local_occlusion(points, ratio=ratio, min_points=cfg.get("occlusion_min_points", 32))
    return points.astype(np.float32)


def apply_vote_augmentation(points, cfg=None):
    cfg = cfg or {}
    points = random_scale(points, scale_low=cfg.get("scale_low", 0.98), scale_high=cfg.get("scale_high", 1.02))
    points = random_rotation_perturb(points, max_degrees=cfg.get("rotation_perturb_deg", 5.0))
    points = random_rotate_z(points, max_degrees=cfg.get("z_rotate_max_deg", 10.0))
    points = random_translate(points, shift_range=cfg.get("translate_range", 0.02))
    points = jitter_points(points, sigma=cfg.get("jitter_sigma", 0.003), clip=cfg.get("jitter_clip", 0.01))
    return points.astype(np.float32)


def load_h5_file(path):
    with h5py.File(path, "r") as handle:
        data = handle["data"][:].astype("float32")
        label_key = "label" if "label" in handle else "labels"
        labels = handle[label_key][:].astype("int64").reshape(-1)
    return data, labels


def read_class_names(path):
    path = Path(path)
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BasePointCloudDataset(Dataset):
    def __init__(self, points, labels, num_points=1024, train=True, normalize=True, augment=False, augment_cfg=None):
        self.points = points
        self.labels = labels
        self.num_points = num_points
        self.train = train
        self.normalize = normalize
        self.augment = augment
        self.augment_cfg = augment_cfg or {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = self.points[index][:, :3].copy()
        label = int(self.labels[index])
        points = random_sample_points(points, self.num_points)
        if self.normalize:
            points = normalize_point_cloud(points)
        if self.train and self.augment:
            points = apply_training_augmentation(points, self.augment_cfg)
        return torch.from_numpy(points).float(), torch.tensor(label, dtype=torch.long)
