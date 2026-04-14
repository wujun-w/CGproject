# Demo Code Walkthrough

This note is designed for the report section `Run Demo code`.

## 1. Data loading
- The training script reads H5 files from `ModelNet40` or `ScanObjectNN`.
- Each sample is a point cloud and a category label.
- The loader keeps only the first three channels `(x, y, z)`.

## 2. Point sampling and normalization
- Each sample is resampled to `1024` points.
- The point cloud is centered by subtracting the centroid.
- The point cloud is scaled to fit a unit sphere.
- During training, optional augmentation adds dropout, rotation, translation, and jitter.

## 3. Forward propagation
- `PointNet` extracts per-point features with shared MLP layers and aggregates them with max pooling.
- `DGCNN` builds a local KNN graph and applies `EdgeConv` repeatedly to capture neighborhood geometry.
- The final global feature is passed into fully connected layers to predict the class label.

## 4. Loss function
- The default loss is cross-entropy classification loss.
- `PointNet` can optionally add a feature transform regularization term.

## 5. Backpropagation and optimization
- The optimizer is `AdamW`.
- For each mini-batch:
  - zero the gradients
  - compute logits
  - compute loss
  - backpropagate the gradients
  - update the weights

## 6. Validation and checkpointing
- After each epoch, the script evaluates on the test split.
- It records validation loss, overall accuracy, and mean per-class accuracy.
- The best checkpoint is saved separately from the latest checkpoint.

## 7. Result export
- Training logs are stored as CSV files.
- Best evaluation metrics are stored as JSON files.
- Per-class accuracy and confusion matrices are exported for direct use in the report.

## 8. Robustness extension
- The robustness script applies four controlled corruptions:
  - point dropout
  - Gaussian noise
  - rotation
  - partial crop
