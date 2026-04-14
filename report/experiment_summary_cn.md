# 点云分类实验结果汇总

## 1. ModelNet40 主实验

### 1.1 训练阶段最佳结果

| Model | Best Epoch | Overall Accuracy | Mean Class Accuracy | Total Training Time (min) |
| --- | ---: | ---: | ---: | ---: |
| PointNet | 92 | 0.8456 | 0.7821 | 15.97 |
| DGCNN | 95 | 0.8756 | 0.8273 | 142.00 |

结论：
- DGCNN 在 ModelNet40 上明显优于 PointNet。
- DGCNN 的 overall accuracy 比 PointNet 高约 3.00 个百分点。
- DGCNN 的 mean class accuracy 比 PointNet 高约 4.51 个百分点。
- 代价是训练时间显著上升，约为 PointNet 的 8.9 倍。

### 1.2 best.pt 正式评估结果

| Model | Dataset | Overall Accuracy | Mean Class Accuracy | Num Samples |
| --- | --- | ---: | ---: | ---: |
| PointNet | ModelNet40 | 0.8472 | 0.7792 | 2468 |
| DGCNN | ModelNet40 | 0.8687 | 0.8198 | 2468 |

结论：
- 使用 best.pt 重新评估后，两种模型的结果与训练阶段记录基本一致。
- 正式报告中应优先引用 best checkpoint 的评估结果，而不是最后一个 epoch 的结果。

### 1.3 ModelNet40 per-class accuracy 观察

PointNet 表现最好的类别：
- airplane: 1.0000
- laptop: 1.0000
- chair: 0.9800
- toilet: 0.9800
- car: 0.9700

PointNet 表现最差的类别：
- flower_pot: 0.0000
- radio: 0.1500
- curtain: 0.5500
- stool: 0.5500
- sink: 0.5500

DGCNN 表现最好的类别：
- airplane: 1.0000
- bowl: 1.0000
- cone: 1.0000
- laptop: 1.0000
- toilet: 1.0000

DGCNN 表现最差的类别：
- flower_pot: 0.1000
- radio: 0.4500
- curtain: 0.5500
- night_stand: 0.5930
- xbox: 0.6000

结论：
- 两个模型都对结构清晰、几何特征稳定的类别表现较好，例如 airplane、laptop、toilet。
- 两个模型都在 flower_pot、radio、curtain 等类别上较弱，说明这些类别更容易受到形状相似性和局部细节不足的影响。
- DGCNN 在困难类别上的表现整体优于 PointNet，但仍然存在明显误分类。

### 1.4 ModelNet40 confusion matrix 典型误分类

PointNet 典型误分类对：
- table -> desk: 20
- night_stand -> dresser: 12
- dresser -> night_stand: 12
- tv_stand -> bookshelf: 10
- flower_pot -> plant: 10
- vase -> cup: 9

DGCNN 典型误分类对：
- night_stand -> dresser: 27
- table -> desk: 13
- glass_box -> dresser: 11
- flower_pot -> plant: 11
- tv_stand -> bookshelf: 10
- plant -> flower_pot: 8

结论：
- 错误主要集中在几何结构相似的类别之间。
- 这类错误非常适合放进 failure case analysis，能很好说明点云分类的类间相似性问题。

## 2. ScanObjectNN 泛化实验

### 2.1 训练阶段最佳结果

| Model | Best Epoch | Overall Accuracy | Mean Class Accuracy | Total Training Time (min) |
| --- | ---: | ---: | ---: | ---: |
| PointNet | 95 | 0.4698 | 0.3960 | 17.29 |
| DGCNN | 97 | 0.6207 | 0.5696 | 162.98 |

### 2.2 best.pt 正式评估结果

| Model | Dataset | Overall Accuracy | Mean Class Accuracy |
| --- | --- | ---: | ---: |
| PointNet | ScanObjectNN | 0.4632 | 0.3847 |
| DGCNN | ScanObjectNN | 0.6149 | 0.5631 |

结论：
- ScanObjectNN 上的准确率明显低于 ModelNet40，说明真实扫描点云更具挑战性。
- DGCNN 在 ScanObjectNN 上依然显著优于 PointNet。
- 这说明局部结构建模对真实、噪声更强、遮挡更多的点云数据更有帮助。

### 2.3 泛化分析

从 ModelNet40 到 ScanObjectNN，两个模型都出现了明显性能下降：
- PointNet overall accuracy 从 0.8472 降到 0.4632。
- DGCNN overall accuracy 从 0.8687 降到 0.6149。

结论：
- 数据域差异非常明显。
- ModelNet40 更接近干净的 CAD 模型，而 ScanObjectNN 更接近真实场景扫描。
- DGCNN 的下降幅度相对更可控，说明其泛化能力更强。

## 3. Advanced Requirement: Robustness 实验

### 3.1 PointNet robustness

| Corruption | Severity 1 | Severity 2 | Severity 3 |
| --- | ---: | ---: | ---: |
| Dropout | 0.8456 | 0.8432 | 0.8436 |
| Noise | 0.8404 | 0.8246 | 0.7998 |
| Rotation | 0.8428 | 0.8460 | 0.8501 |
| Crop | 0.5900 | 0.4254 | 0.1629 |

### 3.2 DGCNN robustness

| Corruption | Severity 1 | Severity 2 | Severity 3 |
| --- | ---: | ---: | ---: |
| Dropout | 0.8659 | 0.8051 | 0.5417 |
| Noise | 0.8740 | 0.8521 | 0.7869 |
| Rotation | 0.8740 | 0.8695 | 0.8703 |
| Crop | 0.4575 | 0.3177 | 0.1418 |

### 3.3 robustness 结论

- 两个模型对 rotation 都较稳定，说明当前训练设置对旋转扰动相对不敏感。
- noise 会逐步降低准确率，但 DGCNN 整体仍优于 PointNet。
- crop 是最致命的扰动，两种模型在严重裁剪下都明显失效。
- 在 dropout 上，轻度扰动时 DGCNN 更强，但在高强度 dropout 下下降更快。

建议在报告中将 advanced requirement 写成：
- 在标准分类任务之外，引入可控 corruption 分析模型鲁棒性。
- 结果表明 DGCNN 在 noise 和 rotation 下更稳健，但两种模型都难以处理严重 crop。
- 后续改进可考虑加入更针对性的局部遮挡增强、点云补全或更强的局部上下文建模方法。

## 4. 报告可直接使用的总总结

- PointNet 适合作为 baseline，结构简单、训练快、复现成本低。
- DGCNN 在 ModelNet40 和 ScanObjectNN 上都优于 PointNet，说明动态图局部特征对点云分类有明显帮助。
- ScanObjectNN 的性能明显更低，验证了真实扫描点云上的泛化挑战。
- 失败案例主要集中在形状相似类别之间，例如 table / desk、dresser / night_stand、flower_pot / plant。
- robustness 实验表明 severe crop 是当前方法最明显的弱点，这可以作为项目的局限分析与未来工作方向。
