# ScanObjectNN 升级实验命令

以下命令默认在已激活 `torchcuda` 环境后执行。

## 1. PointNet enhanced

```powershell
python source/train.py `
  --model pointnet `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --experiment pointnet_scanobjectnn_upgrade `
  --epochs 200 `
  --batch-size 32 `
  --num-workers 0 `
  --num-points 2048 `
  --feature-transform `
  --lr 5e-4 `
  --weight-decay 1e-4 `
  --occlusion-prob 0.4 `
  --occlusion-min-ratio 0.1 `
  --occlusion-max-ratio 0.2 `
  --scale-low 0.9 `
  --scale-high 1.1 `
  --rotation-perturb-deg 10 `
  --device cuda
```

## 2. DGCNN enhanced

```powershell
python source/train.py `
  --model dgcnn `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --experiment dgcnn_scanobjectnn_upgrade `
  --epochs 200 `
  --batch-size 12 `
  --num-workers 0 `
  --num-points 2048 `
  --k 40 `
  --lr 5e-4 `
  --weight-decay 1e-4 `
  --occlusion-prob 0.4 `
  --occlusion-min-ratio 0.1 `
  --occlusion-max-ratio 0.2 `
  --scale-low 0.9 `
  --scale-high 1.1 `
  --rotation-perturb-deg 10 `
  --device cuda
```

## 3. best.pt 正式评估 + voting

PointNet:

```powershell
python source/evaluate.py `
  --model pointnet `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --checkpoint checkpoints/pointnet_scanobjectnn_upgrade/best.pt `
  --experiment pointnet_scanobjectnn_upgrade_eval `
  --batch-size 32 `
  --num-workers 0 `
  --num-points 2048 `
  --vote-num 10 `
  --vote-aug `
  --device cuda
```

DGCNN:

```powershell
python source/evaluate.py `
  --model dgcnn `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --checkpoint checkpoints/dgcnn_scanobjectnn_upgrade/best.pt `
  --experiment dgcnn_scanobjectnn_upgrade_eval `
  --batch-size 12 `
  --num-workers 0 `
  --num-points 2048 `
  --vote-num 10 `
  --vote-aug `
  --device cuda
```

## 4. Crop robustness before vs after

旧版 DGCNN checkpoint:

```powershell
python source/robustness.py `
  --model dgcnn `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --checkpoint checkpoints/dgcnn_scanobjectnn/best.pt `
  --experiment dgcnn_scanobjectnn_crop_before `
  --batch-size 16 `
  --num-workers 0 `
  --num-points 1024 `
  --vote-num 10 `
  --crop-mode local_occlusion `
  --device cuda
```

增强版 DGCNN checkpoint:

```powershell
python source/robustness.py `
  --model dgcnn `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --checkpoint checkpoints/dgcnn_scanobjectnn_upgrade/best.pt `
  --experiment dgcnn_scanobjectnn_crop_after `
  --batch-size 12 `
  --num-workers 0 `
  --num-points 2048 `
  --vote-num 10 `
  --crop-mode local_occlusion `
  --device cuda
```

## 5. DGCNN ablation

仅 `2048 points`:

```powershell
python source/train.py `
  --model dgcnn `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --experiment dgcnn_scanobjectnn_ablate_2048 `
  --epochs 200 `
  --batch-size 12 `
  --num-workers 0 `
  --num-points 2048 `
  --lr 5e-4 `
  --disable-legacy-dropout `
  --occlusion-prob 0.0 `
  --device cuda
```

`2048 + k=40`:

```powershell
python source/train.py `
  --model dgcnn `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --experiment dgcnn_scanobjectnn_ablate_2048_k40 `
  --epochs 200 `
  --batch-size 12 `
  --num-workers 0 `
  --num-points 2048 `
  --k 40 `
  --lr 5e-4 `
  --disable-legacy-dropout `
  --occlusion-prob 0.0 `
  --device cuda
```

`2048 + k=40 + new augmentation`:

```powershell
python source/train.py `
  --model dgcnn `
  --dataset scanobjectnn `
  --data-root datasets/scanobjectnn/main_split `
  --experiment dgcnn_scanobjectnn_ablate_aug `
  --epochs 200 `
  --batch-size 12 `
  --num-workers 0 `
  --num-points 2048 `
  --k 40 `
  --lr 5e-4 `
  --occlusion-prob 0.4 `
  --occlusion-min-ratio 0.1 `
  --occlusion-max-ratio 0.2 `
  --scale-low 0.9 `
  --scale-high 1.1 `
  --rotation-perturb-deg 10 `
  --device cuda
```

`2048 + k=40 + new augmentation + voting`:
- 训练同上一条
- 评估时使用 `--vote-num 10 --vote-aug`
