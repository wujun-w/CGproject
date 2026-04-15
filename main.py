import argparse
from pathlib import Path

from project.config import EXPERIMENTS
from project.engine.trainer import run_experiment, run_test_only


def parse_args():
    # 课程项目只保留一个统一入口，避免为四组实验分别维护脚本。
    parser = argparse.ArgumentParser(description="点云分类课程项目统一入口")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=sorted(EXPERIMENTS.keys()),
        help="实验名称",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="运行模式：train 表示训练并评测，test 表示仅加载最佳权重做评测",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="测试模式下可选，指定权重路径；为空时默认读取结果目录中的 best_model.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 训练模式会完整执行训练、验证、保存最佳权重和最终测试。
    if args.mode == "train":
        run_experiment(args.exp)
        return

    # 测试模式默认读取当前实验目录下的最佳权重。
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    run_test_only(args.exp, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()
