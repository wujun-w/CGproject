# CG Project: Point Cloud Classification

本项目是 `CS5182` 课程项目的代码仓库，任务方向为**点云分类**。

项目基于两个开源实现完成：

- `PointNet`：<https://github.com/fxia22/pointnet.pytorch>
- `DGCNN`：<https://github.com/WangYueFt/dgcnn>

本仓库在开源实现的基础上，统一了数据读取、训练、评测、日志、错误样本导出和实验汇总流程，用于完成 4 组标准分类实验。

## 项目目标

本项目完成以下 4 组实验：

1. `PointNet` 在 `ModelNet40` 上训练与测试
2. `DGCNN` 在 `ModelNet40` 上训练与测试
3. `PointNet` 在 `ScanObjectNN` 上训练与测试
4. `DGCNN` 在 `ScanObjectNN` 上训练与测试

项目重点放在课程要求中的基础部分：

- 使用开源实现完成点云分类任务
- 在两个数据集上训练并测试模型
- 比较两种方法的分类性能
- 记录训练时间、显存和评测结果
- 导出错误样本并进行失败案例分析

## 仓库内容

本仓库**只包含代码和文档**，不包含：

- 原始数据集
- 训练结果 `results/`
- 本地虚拟环境

## 目录结构

```text
CG_Project/
  main.py
  run_all_experiments.py
  project/
  tools/
  docs/
  dgcnn-master/
  pointnet.pytorch-master/
  requirements.txt
```

说明：

- `main.py`：单组实验统一入口
- `run_all_experiments.py`：按顺序一键运行 4 组实验
- `project/`：课程项目的统一训练框架
- `tools/`：数据转换与实验汇总脚本
- `docs/`：中文使用说明和代码说明
- `dgcnn-master/`：DGCNN 开源实现
- `pointnet.pytorch-master/`：PointNet 开源实现

## 环境要求

建议使用：

- Python `3.10+`
- PyTorch（根据本机或远程服务器 CUDA 环境单独安装）

安装项目依赖：

```bash
pip install -r requirements.txt
```

注意：

- `requirements.txt` 不包含 `torch`
- 请根据实际机器环境自行安装对应版本的 PyTorch

## 数据准备

代码默认使用以下数据目录：

- `data_cg/ModelNet40_h5/modelnet40_ply_hdf5_2048`
- `data_cg/ScanObjectNN/main_split_nobg`

如果 `ModelNet40` 还是原始 `.off` 网格格式，需要先转换为 H5：

```bash
python tools/convert_modelnet40_to_h5.py
```

`ScanObjectNN` 目录下建议包含：

```text
training_objectdataset.h5
test_objectdataset.h5
label_mapping.csv
```

## 运行方式

### 运行单组实验

```bash
python main.py --exp pointnet_modelnet40
python main.py --exp dgcnn_modelnet40
python main.py --exp pointnet_scanobjectnn
python main.py --exp dgcnn_scanobjectnn
```

### 一键顺序运行全部实验

```bash
python run_all_experiments.py
```

该脚本会按固定顺序逐组运行实验，每组实验在独立子进程中执行，结束后再开始下一组，以保证显存可以正常释放。

### 仅执行测试

```bash
python main.py --exp pointnet_modelnet40 --mode test
```

## 结果输出

每组实验的结果默认保存在：

```text
results/<experiment_name>/
```

主要包括：

- `logs/train.log`
- `metrics/history.csv`
- `metrics/final_metrics.json`
- `metrics/per_class_accuracy.csv`
- `metrics/wrong_cases.csv`
- `plots/training_curves.png`
- `checkpoints/best_model.pt`
- `exports/wrong_cases/*.ply`

完成多组实验后，可以执行汇总：

```bash
python tools/summarize_experiments.py
```

汇总结果保存在：

```text
results/summary/
```

## 文档

详细说明见：

- [docs/使用说明.md](docs/使用说明.md)
- [docs/代码说明.md](docs/代码说明.md)

## 说明

本仓库中的控制台提示和代码注释以中文为主；结构化实验结果字段统一使用英文，方便后续统计、汇总和画图。
