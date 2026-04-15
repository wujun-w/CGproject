import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="汇总四组点云分类实验结果")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="实验结果根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/summary",
        help="汇总结果输出目录",
    )
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_training_summary(summary_path):
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_final_metrics(metrics_path):
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def collect_rows(results_root):
    rows = []
    for exp_dir in sorted(Path(results_root).iterdir()):
        if not exp_dir.is_dir():
            continue

        training_summary = load_training_summary(exp_dir / "metrics" / "training_summary.json")
        final_metrics = load_final_metrics(exp_dir / "metrics" / "final_metrics.json")
        if not final_metrics:
            continue

        rows.append(
            {
                "experiment": exp_dir.name,
                "best_epoch": training_summary.get("best_epoch", ""),
                "best_oa": training_summary.get("best_oa", ""),
                "final_test_oa": final_metrics.get("oa", ""),
                "final_test_macc": final_metrics.get("macc", ""),
                "eval_time_sec": final_metrics.get("eval_time_sec", ""),
                "num_samples": final_metrics.get("num_samples", ""),
                "wrong_case_count": final_metrics.get("wrong_case_count", ""),
                "total_train_time_sec": training_summary.get("total_train_time_sec", ""),
                "avg_epoch_time_sec": training_summary.get("avg_epoch_time_sec", ""),
                "avg_eval_time_sec": training_summary.get("avg_eval_time_sec", ""),
                "max_peak_gpu_mem_mb": training_summary.get("max_peak_gpu_mem_mb", ""),
            }
        )
    return rows


def save_csv(path, rows):
    import csv

    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_summary_plot(rows, output_path):
    experiments = [row["experiment"] for row in rows]
    oa_values = [float(row["final_test_oa"] or 0.0) for row in rows]
    macc_values = [float(row["final_test_macc"] or 0.0) for row in rows]
    train_time_values = [float(row["total_train_time_sec"] or 0.0) for row in rows]
    epoch_time_values = [float(row["avg_epoch_time_sec"] or 0.0) for row in rows]
    gpu_mem_values = [float(row["max_peak_gpu_mem_mb"] or 0.0) for row in rows]

    plt.figure(figsize=(14, 10))
    positions = list(range(len(experiments)))
    width = 0.35

    ax1 = plt.subplot(2, 2, 1)
    ax1.bar([item - width / 2 for item in positions], oa_values, width=width, label="OA")
    ax1.bar([item + width / 2 for item in positions], macc_values, width=width, label="mAcc")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(experiments, rotation=20)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("accuracy")
    ax1.set_title("Accuracy Summary")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(positions, train_time_values)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(experiments, rotation=20)
    ax2.set_ylabel("seconds")
    ax2.set_title("Total Train Time")

    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(positions, epoch_time_values)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(experiments, rotation=20)
    ax3.set_ylabel("seconds")
    ax3.set_title("Average Epoch Time")

    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(positions, gpu_mem_values)
    ax4.set_xticks(positions)
    ax4.set_xticklabels(experiments, rotation=20)
    ax4.set_ylabel("MB")
    ax4.set_title("Peak GPU Memory")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = ensure_dir(args.output_dir)

    if not results_root.exists():
        print("结果根目录不存在，请先完成训练。", flush=True)
        return

    rows = collect_rows(results_root)
    if not rows:
        print("未发现可汇总的实验结果，请先至少完成一组实验。", flush=True)
        return

    save_csv(output_dir / "experiment_summary.csv", rows)
    save_json(output_dir / "experiment_summary.json", rows)
    save_summary_plot(rows, output_dir / "experiment_summary.png")
    print(f"实验汇总完成，输出目录：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
