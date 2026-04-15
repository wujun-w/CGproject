from project.models.dgcnn_wrapper import DGCNNWrapper
from project.models.pointnet_wrapper import PointNetWrapper


def build_model(config):
    # 统一模型构建入口，训练层不再关心底层网络细节。
    model_name = config["model_name"]
    num_classes = config["num_classes"]

    if model_name == "pointnet":
        return PointNetWrapper(
            num_classes=num_classes,
            feature_transform=config.get("feature_transform", False),
        )

    if model_name == "dgcnn":
        return DGCNNWrapper(
            num_classes=num_classes,
            emb_dims=config.get("emb_dims", 1024),
            dropout=config.get("dropout", 0.5),
            k=config.get("k", 20),
        )

    raise ValueError(f"Unsupported model: {model_name}")
