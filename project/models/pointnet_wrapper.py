import importlib.util
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


_POINTNET_MODULE = None


def _load_pointnet_module():
    global _POINTNET_MODULE
    if _POINTNET_MODULE is not None:
        return _POINTNET_MODULE

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "pointnet.pytorch-master" / "pointnet" / "model.py"
    spec = importlib.util.spec_from_file_location("cg_project_pointnet_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _POINTNET_MODULE = module
    return _POINTNET_MODULE


class PointNetWrapper(nn.Module):
    def __init__(self, num_classes, feature_transform):
        super().__init__()
        pointnet_module = _load_pointnet_module()
        self.feature_transform = feature_transform
        # 直接复用开源 PointNet 的分类头。
        self.model = pointnet_module.PointNetCls(k=num_classes, feature_transform=feature_transform)
        self.feature_transform_regularizer = pointnet_module.feature_transform_regularizer

    def forward(self, points, labels=None):
        # PointNet 原始实现要求输入为 (B, 3, N)。
        points = points.transpose(2, 1)
        scores, _, trans_feat = self.model(points)

        output = {
            "scores": scores,
            "preds": scores.argmax(dim=1),
        }

        if labels is not None:
            loss = F.nll_loss(scores, labels)
            if self.feature_transform and trans_feat is not None:
                loss = loss + self.feature_transform_regularizer(trans_feat) * 0.001
            output["loss"] = loss

        return output
