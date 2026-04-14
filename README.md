# Point Cloud Shape Classification Course Project

This repository provides a clean PyTorch project scaffold for a course project on point cloud classification with `PointNet` and `DGCNN`.

## Project scope
- Train and compare `PointNet` and `DGCNN` on `ModelNet40`
- Evaluate generalization on `ScanObjectNN`
- Run robustness experiments for the advanced requirement
- Export logs, metrics, confusion matrices, and figures for the report

## Repository layout
- `source/`: training, evaluation, models, datasets, and utilities
- `results/`: training logs, evaluation logs, figures, and exported metrics
- `report/`: Chinese report and presentation outlines

## Recommended environment
Use Conda with the provided `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate pointcloud-cls
```

Install a CUDA-enabled PyTorch build that matches your local CUDA version if needed.

## Dataset preparation
The code expects the following layout:

```text
datasets/
  modelnet40_ply_hdf5_2048/
    ply_data_train*.h5
    ply_data_test*.h5
    shape_names.txt
  scanobjectnn/
    main_split/
      training_objectdataset_augmentedrot_scale75.h5
      test_objectdataset_augmentedrot_scale75.h5
      classes.txt
```

If your ScanObjectNN files use different names, pass the paths explicitly to the training and evaluation commands.

## Training
Train PointNet on ModelNet40:

```powershell
python source/train.py `
  --model pointnet `
  --dataset modelnet40 `
  --data-root datasets/modelnet40_ply_hdf5_2048 `
  --experiment pointnet_modelnet40
```

Train DGCNN on ModelNet40:

```powershell
python source/train.py `
  --model dgcnn `
  --dataset modelnet40 `
  --data-root datasets/modelnet40_ply_hdf5_2048 `
  --experiment dgcnn_modelnet40
```

## Evaluation
```powershell
python source/evaluate.py `
  --model dgcnn `
  --dataset modelnet40 `
  --data-root datasets/modelnet40_ply_hdf5_2048 `
  --checkpoint checkpoints/dgcnn_modelnet40/best.pt `
  --experiment dgcnn_modelnet40_eval
```

## Robustness experiment
```powershell
python source/robustness.py `
  --model dgcnn `
  --dataset modelnet40 `
  --data-root datasets/modelnet40_ply_hdf5_2048 `
  --checkpoint checkpoints/dgcnn_modelnet40/best.pt `
  --experiment dgcnn_modelnet40_robustness
```

See `source/scripts/demo_walkthrough.md` for a report-friendly explanation of each pipeline step.
