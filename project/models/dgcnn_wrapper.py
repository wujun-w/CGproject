import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


_DGCNN_MODULE = None


def _load_dgcnn_module():
    global _DGCNN_MODULE
    if _DGCNN_MODULE is not None:
        return _DGCNN_MODULE

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "dgcnn-master" / "pytorch" / "model.py"
    spec = importlib.util.spec_from_file_location("cg_project_dgcnn_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 原仓库把 device 硬编码成了 cuda，这里做最小修补，跟随输入张量设备即可。
    def get_graph_feature(x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = module.knn(x, k=k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)

        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    module.get_graph_feature = get_graph_feature
    _DGCNN_MODULE = module
    return _DGCNN_MODULE


class DGCNNWrapper(nn.Module):
    def __init__(self, num_classes, emb_dims, dropout, k):
        super().__init__()
        dgcnn_module = _load_dgcnn_module()
        # 原仓库需要一个轻量参数对象，这里只保留当前实验用到的字段。
        args = SimpleNamespace(emb_dims=emb_dims, dropout=dropout, k=k)
        self.model = dgcnn_module.DGCNN(args=args, output_channels=num_classes)

    def forward(self, points, labels=None):
        # DGCNN 原始实现要求输入为 (B, 3, N)。
        points = points.transpose(2, 1)
        scores = self.model(points)

        output = {
            "scores": scores,
            "preds": scores.argmax(dim=1),
        }

        if labels is not None:
            output["loss"] = F.cross_entropy(scores, labels)

        return output
