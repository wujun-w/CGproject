import numpy as np


def sample_points(points, num_points, training):
    # 点数不足时允许重复采样；点数过多时训练集随机采样，测试集截取前 num_points 个。
    total_points = points.shape[0]
    if total_points == num_points:
        return points.copy()

    if total_points > num_points:
        if training:
            choice = np.random.choice(total_points, num_points, replace=False)
            return points[choice]
        return points[:num_points].copy()

    choice = np.random.choice(total_points, num_points, replace=True)
    return points[choice]


def normalize_pointcloud(points):
    # 统一平移到原点，并缩放到单位球内。
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if radius > 0:
        centered = centered / radius
    return centered.astype(np.float32)


def random_rotate_z(points):
    # 分类任务只做绕 z 轴旋转，保持增强逻辑简单。
    angle = np.random.uniform(0.0, np.pi * 2.0)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation = np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return points @ rotation.T


def random_jitter(points, sigma=0.01, clip=0.02):
    # 加入轻微高斯噪声，模拟测量扰动。
    noise = np.clip(np.random.normal(0.0, sigma, size=points.shape), -clip, clip)
    return (points + noise).astype(np.float32)


def build_points(points, num_points, training):
    # 统一封装训练/测试阶段的最小数据处理流程。
    sampled = sample_points(points, num_points=num_points, training=training)
    sampled = normalize_pointcloud(sampled)

    if training:
        sampled = random_rotate_z(sampled)
        sampled = random_jitter(sampled)

    return sampled.astype(np.float32)
