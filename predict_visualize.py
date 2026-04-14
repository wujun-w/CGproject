import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import torch
import numpy as np
import importlib
from data_utils.ModelNetDataLoader import ModelNetDataLoader

# 设定你要导出的错误案例数量
MAX_ERROR_CASES = 5
OUTPUT_DIR = 'error_visualizations'
MODEL_NAME = 'pointnet_cls' # 我们用最开始跑得很快那个基础版模型来找错题

def save_point_cloud(points, filename):
    """保存为 .xyz 格式，可以直接用 MeshLab 打开"""
    with open(filename, 'w') as f:
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def main():
    # 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading testing dataset...")
    class DummyArgs:
        use_uniform_sample = False
        use_normals = False
        num_category = 40
        process_data = False
        num_point = 1024
    args = DummyArgs()
    test_dataset = ModelNetDataLoader(root='data/modelnet40_normal_resampled', args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 类别名字映射字典（你需要检查你的数据集是不是刚好对应这些下标，这40个是ModelNet标准名称）
    with open('data/modelnet40_normal_resampled/modelnet40_shape_names.txt', 'r') as f:
        shape_names = [line.rstrip() for line in f]
    
    print(f"Loading {MODEL_NAME} model...")
    # 这个路径是你之前跑出来 90.5% 的基础版点云权重的路径
    model_path = os.path.join('log', 'classification', 'pointnet_cls_msg', 'checkpoints', 'best_model.pth')
    
    # 动态加载模型架构
    model_module = importlib.import_module(MODEL_NAME)
    classifier = model_module.get_model(40, normal_channel=False).cuda()
    
    # 解决 "Weights only load failed" 报错，强行加载
    checkpoint = torch.load(model_path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    error_count = 0
    print("Start finding error cases...")
    
    with torch.no_grad():
        for i, (points, target) in enumerate(testDataLoader):
            
            # [1, 1024, 3] 转成 [1, 3, 1024] 给模型
            points_tensor = points.transpose(2, 1).cuda()
            target_tensor = target.cuda()
            
            # 前向传播
            pred, _ = classifier(points_tensor)
            pred_choice = pred.data.max(1)[1] # 获取最大概率的分类下标
            
            # 如果预测错了！
            if pred_choice.item() != target_tensor.item():
                true_label = shape_names[target_tensor.item()]
                pred_label = shape_names[pred_choice.item()]
                
                print(f"Error #{error_count+1}: True -> [{true_label}], but Predicted as -> [{pred_label}]")
                
                # 重新把点提取出来到 CPU 上转成 numpy 画图 (注意要把维度转回 [1024, 3])
                points_np = points[0].numpy()
                
                # 导出文件名为: 真实标签_认成了_预测标签.xyz
                filename = os.path.join(OUTPUT_DIR, f"error{error_count+1}_True_{true_label}_Pred_{pred_label}.xyz")
                save_point_cloud(points_np, filename)
                
                error_count += 1
                if error_count >= MAX_ERROR_CASES:
                    break
    
    print(f"\nDone! Exported {error_count} error cases to the '{OUTPUT_DIR}' folder.")
    print("Now you can open these .xyz files using MeshLab and screenshot them for your report!")

if __name__ == '__main__':
    main()
