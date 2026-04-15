import subprocess
import sys
import time
from pathlib import Path


# 按实验计划固定顺序依次执行 4 组主实验。
EXPERIMENT_ORDER = [
    "pointnet_modelnet40",
    "dgcnn_modelnet40",
    "pointnet_scanobjectnn",
    "dgcnn_scanobjectnn",
]

# 两组实验之间留一个短暂停顿，确保上一组子进程完全退出。
COOLDOWN_SECONDS = 5


def run_single_experiment(project_root, exp_name, exp_index, total_count):
    command = [sys.executable, "main.py", "--exp", exp_name]
    print(
        f"\n========== 开始执行第 {exp_index}/{total_count} 组实验：{exp_name} ==========",
        flush=True,
    )
    print(f"执行命令：{' '.join(command)}", flush=True)

    # 每组实验都放在独立子进程中运行。子进程退出后，训练进程占用的显存会随 CUDA 上下文一起释放。
    completed = subprocess.run(command, cwd=project_root)
    if completed.returncode != 0:
        print(f"实验执行失败：{exp_name}，返回码={completed.returncode}", flush=True)
        return completed.returncode

    print(f"实验执行完成：{exp_name}", flush=True)
    print(f"等待 {COOLDOWN_SECONDS} 秒，确保上一组实验进程完全退出并释放显存。", flush=True)
    time.sleep(COOLDOWN_SECONDS)
    return 0


def main():
    project_root = Path(__file__).resolve().parent
    total_count = len(EXPERIMENT_ORDER)

    print("开始顺序执行全部实验。", flush=True)
    print(f"项目目录：{project_root}", flush=True)
    print(f"当前 Python：{sys.executable}", flush=True)
    print("实验将严格按顺序执行，中途若某一组失败，脚本会立即停止。", flush=True)

    for exp_index, exp_name in enumerate(EXPERIMENT_ORDER, start=1):
        return_code = run_single_experiment(
            project_root=project_root,
            exp_name=exp_name,
            exp_index=exp_index,
            total_count=total_count,
        )
        if return_code != 0:
            sys.exit(return_code)

    print("\n全部实验执行完成。", flush=True)
    print("你现在可以运行汇总脚本：python tools/summarize_experiments.py", flush=True)


if __name__ == "__main__":
    main()
