import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.config import EXPERIMENTS
from project.datasets import build_dataset
from project.engine.evaluator import evaluate_model
from project.models.builder import build_model
from project.utils.export import save_history_plot
from project.utils.io import (
    make_experiment_dirs,
    sanitize_for_json,
    save_csv,
    save_json,
    save_model_checkpoint,
)
from project.utils.logger import SimpleLogger
from project.utils.metrics import compute_classification_metrics


def set_seed(seed):
    # 固定随机种子，保证本地调试和远程正式训练行为尽量一致。
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_optimizer(model, config):
    optimizer_name = config["optimizer_name"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer, config):
    scheduler_name = config["scheduler_name"]

    if scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=config.get("step_size", 20),
            gamma=config.get("gamma", 0.5),
        )

    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=config["lr"] * 0.01,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_one_epoch(model, dataloader, optimizer, device, class_names, desc):
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    epoch_start = time.time()
    progress = tqdm(dataloader, desc=desc, leave=False, ncols=120)

    for points, labels, _ in progress:
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(points, labels=labels)
        loss = output["loss"]
        preds = output["preds"]

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)
        batch_oa = float((preds.detach().cpu() == labels.detach().cpu()).float().mean().item())
        progress.set_postfix({"loss": f"{avg_loss:.4f}", "oa": f"{batch_oa:.4f}"})

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_classification_metrics(
        preds=all_preds,
        labels=all_labels,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["epoch_time_sec"] = time.time() - epoch_start
    metrics["num_samples"] = int(total_samples)
    return metrics


def prepare_loaders(config):
    # 训练集和测试集都走统一的数据接口，避免为两个模型分别写脚本。
    train_dataset = build_dataset(
        dataset_name=config["dataset_name"],
        data_root=config["data_root"],
        split="train",
        num_points=config["num_points"],
        training=True,
    )
    test_dataset = build_dataset(
        dataset_name=config["dataset_name"],
        data_root=config["data_root"],
        split="test",
        num_points=config["num_points"],
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def prepare_test_loader(config):
    # 仅测试时不再构建训练集，减少无意义的数据加载开销。
    test_dataset = build_dataset(
        dataset_name=config["dataset_name"],
        data_root=config["data_root"],
        split="test",
        num_points=config["num_points"],
        training=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )
    return test_dataset, test_loader


def log_epoch(logger, epoch_index, train_metrics, test_metrics, peak_memory_mb):
    logger.log(
        "第 {epoch:03d} 轮训练完成 | "
        "训练损失: {train_loss:.4f} | 训练 OA: {train_oa:.4f} | 训练 mAcc: {train_macc:.4f} | "
        "测试损失: {test_loss:.4f} | 测试 OA: {test_oa:.4f} | 测试 mAcc: {test_macc:.4f} | "
        "训练耗时: {train_time:.2f}s | 测试耗时: {test_time:.2f}s | 峰值显存: {peak_mem:.2f} MB".format(
            epoch=epoch_index,
            train_loss=train_metrics["loss"],
            train_oa=train_metrics["oa"],
            train_macc=train_metrics["macc"],
            test_loss=test_metrics["loss"],
            test_oa=test_metrics["oa"],
            test_macc=test_metrics["macc"],
            train_time=train_metrics["epoch_time_sec"],
            test_time=test_metrics["eval_time_sec"],
            peak_mem=peak_memory_mb,
        )
    )


def build_history_row(epoch_index, train_metrics, test_metrics, peak_memory_mb, current_lr):
    return {
        "epoch": epoch_index,
        "train_loss": train_metrics["loss"],
        "train_oa": train_metrics["oa"],
        "train_macc": train_metrics["macc"],
        "test_loss": test_metrics["loss"],
        "test_oa": test_metrics["oa"],
        "test_macc": test_metrics["macc"],
        "epoch_time_sec": train_metrics["epoch_time_sec"],
        "eval_time_sec": test_metrics["eval_time_sec"],
        "peak_gpu_mem_mb": peak_memory_mb,
        "lr": current_lr,
    }


def run_experiment(exp_name):
    config = copy.deepcopy(EXPERIMENTS[exp_name])
    result_dirs = make_experiment_dirs(config["results_root"], exp_name)
    logger = SimpleLogger(result_dirs["logs"] / "train.log")

    save_json(result_dirs["metrics"] / "config.json", sanitize_for_json(config))
    logger.log(f"开始执行实验：{exp_name}")
    logger.log(f"结果目录：{result_dirs['root']}")

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"当前设备：{device}")
    if device.type == "cuda":
        logger.log(f"GPU 名称：{torch.cuda.get_device_name(device)}")

    train_dataset, test_dataset, train_loader, test_loader = prepare_loaders(config)
    class_names = test_dataset.class_names
    logger.log(
        f"数据集已加载 | 训练样本数: {len(train_dataset)} | 测试样本数: {len(test_dataset)} | 类别数: {len(class_names)}"
    )

    model = build_model(config).to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    history = []
    best_oa = -1.0
    best_epoch = 0
    total_start_time = time.time()

    # 每一轮都保存结构化历史数据，后续画图和写报告直接读取即可。
    for epoch_index in range(1, config["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            class_names=class_names,
            desc=f"训练中 第 {epoch_index}/{config['epochs']} 轮",
        )
        test_metrics, _ = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            desc=f"评测中 第 {epoch_index}/{config['epochs']} 轮",
            export_dir=None,
            experiment_name=exp_name,
            dataset_name=config["dataset_name"],
            model_name=config["model_name"],
        )

        peak_memory_mb = 0.0
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        current_lr = optimizer.param_groups[0]["lr"]
        history_row = build_history_row(
            epoch_index=epoch_index,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            peak_memory_mb=peak_memory_mb,
            current_lr=current_lr,
        )
        history.append(history_row)

        save_csv(result_dirs["metrics"] / "history.csv", history)
        log_epoch(logger, epoch_index, train_metrics, test_metrics, peak_memory_mb)

        checkpoint_payload = {
            "epoch": epoch_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": sanitize_for_json(config),
            "test_oa": test_metrics["oa"],
        }
        save_model_checkpoint(result_dirs["checkpoints"] / "latest_model.pt", checkpoint_payload)

        if test_metrics["oa"] >= best_oa:
            best_oa = test_metrics["oa"]
            best_epoch = epoch_index
            save_model_checkpoint(result_dirs["checkpoints"] / "best_model.pt", checkpoint_payload)
            logger.log(f"已更新最佳模型：第 {epoch_index} 轮 | 最佳测试 OA: {best_oa:.4f}")

        scheduler.step()

    total_train_time = time.time() - total_start_time
    logger.log(f"训练阶段结束，总耗时：{total_train_time:.2f} 秒")

    # 训练结束后固定保存曲线图，避免后续手工整理数据时重复画图。
    save_history_plot(history, result_dirs["plots"] / "training_curves.png", exp_name=exp_name)
    logger.log("训练曲线图片已保存")

    best_checkpoint_path = result_dirs["checkpoints"] / "best_model.pt"
    final_metrics = run_test_only(exp_name, checkpoint_path=best_checkpoint_path, logger=logger)

    summary_payload = {
        "experiment": exp_name,
        "best_epoch": best_epoch,
        "best_oa": best_oa,
        "total_train_time_sec": total_train_time,
        "avg_epoch_time_sec": float(np.mean([item["epoch_time_sec"] for item in history])),
        "avg_eval_time_sec": float(np.mean([item["eval_time_sec"] for item in history])),
        "max_peak_gpu_mem_mb": float(np.max([item["peak_gpu_mem_mb"] for item in history])),
        "final_test_oa": final_metrics["oa"],
        "final_test_macc": final_metrics["macc"],
        "final_wrong_case_count": final_metrics["wrong_case_count"],
    }
    save_json(result_dirs["metrics"] / "training_summary.json", summary_payload)
    logger.log("实验流程全部完成")
    logger.close()


def run_test_only(exp_name, checkpoint_path=None, logger=None):
    config = copy.deepcopy(EXPERIMENTS[exp_name])
    result_dirs = make_experiment_dirs(config["results_root"], exp_name)
    created_logger = False

    if logger is None:
        logger = SimpleLogger(result_dirs["logs"] / "test.log")
        created_logger = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset, test_loader = prepare_test_loader(config)
    class_names = test_dataset.class_names

    model = build_model(config).to(device)
    if checkpoint_path is None:
        checkpoint_path = result_dirs["checkpoints"] / "best_model.pt"
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.log(f"已加载权重：{checkpoint_path}")

    final_metrics, wrong_cases = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        desc="最终评测中",
        export_dir=result_dirs["exports"] / "wrong_cases",
        experiment_name=exp_name,
        dataset_name=config["dataset_name"],
        model_name=config["model_name"],
    )

    per_class_rows = [
        {"class_name": class_name, "class_accuracy": class_accuracy}
        for class_name, class_accuracy in final_metrics["per_class_acc"].items()
    ]

    # 指标文件统一使用英文字段，便于后续汇总和画图。
    save_csv(result_dirs["metrics"] / "per_class_accuracy.csv", per_class_rows)
    save_csv(result_dirs["metrics"] / "wrong_cases.csv", wrong_cases)
    save_json(result_dirs["metrics"] / "final_metrics.json", final_metrics)

    logger.log(
        "最终测试完成 | 测试 OA: {oa:.4f} | 测试 mAcc: {macc:.4f} | 测试损失: {loss:.4f} | "
        "测试耗时: {eval_time:.2f}s | 错误样本数: {wrong_case_count}".format(
            oa=final_metrics["oa"],
            macc=final_metrics["macc"],
            loss=final_metrics["loss"],
            eval_time=final_metrics["eval_time_sec"],
            wrong_case_count=final_metrics["wrong_case_count"],
        )
    )

    if created_logger:
        logger.close()

    return final_metrics
