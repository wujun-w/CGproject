from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.utils.io import ensure_dir


def save_history_plot(history, plot_path, exp_name):
    # 将训练过程中的关键数值集中画到一张图里，便于后续直接插入报告。
    plot_path = Path(plot_path)
    ensure_dir(plot_path.parent)

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    test_loss = [item["test_loss"] for item in history]
    train_oa = [item["train_oa"] for item in history]
    test_oa = [item["test_oa"] for item in history]
    test_macc = [item["test_macc"] for item in history]
    epoch_time = [item["epoch_time_sec"] for item in history]
    peak_mem = [item["peak_gpu_mem_mb"] for item in history]

    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, test_loss, label="test_loss")
    ax1.set_title(f"{exp_name} loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, train_oa, label="train_oa")
    ax2.plot(epochs, test_oa, label="test_oa")
    ax2.plot(epochs, test_macc, label="test_macc")
    ax2.set_title(f"{exp_name} accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs, epoch_time, label="epoch_time_sec")
    ax3.set_title(f"{exp_name} epoch time")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("seconds")
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(epochs, peak_mem, label="peak_gpu_mem_mb")
    ax4.set_title(f"{exp_name} gpu memory")
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("MB")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_wrong_case_points(points, export_dir, row):
    export_dir = ensure_dir(export_dir)
    file_name = (
        f"{row['sample_index']:05d}_"
        f"{row['shape_id']}_"
        f"true_{row['true_name']}_pred_{row['pred_name']}.ply"
    )
    file_path = export_dir / file_name
    write_ply(points, file_path)


def write_ply(points, file_path):
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    points = np.asarray(points, dtype=np.float32)

    # 直接输出 ASCII PLY，便于用 MeshLab 打开查看错误样本。
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
