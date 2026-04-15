import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="将 ModelNet40 的 OFF 网格转换为 H5 点云文件。")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data_cg/ModelNet40/ModelNet40"),
        help="原始 ModelNet40 类别目录根路径。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_cg/ModelNet40_h5/modelnet40_ply_hdf5_2048"),
        help="H5 输出目录。",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="每个样本采样的点数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证转换结果可复现。",
    )
    return parser.parse_args()


def next_data_line(lines, start_index):
    index = start_index
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line and not line.startswith("#"):
            return line, index
    raise ValueError("Unexpected end of OFF file.")


def load_off_mesh(path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    header, index = next_data_line(lines, 0)
    if header.startswith("OFF"):
        counts_line = header[3:].strip()
        if not counts_line:
            counts_line, index = next_data_line(lines, index)
    else:
        raise ValueError(f"{path} is not a valid OFF file.")

    vertex_count, face_count, _ = map(int, counts_line.split()[:3])

    vertices = np.empty((vertex_count, 3), dtype=np.float64)
    for vertex_idx in range(vertex_count):
        line, index = next_data_line(lines, index)
        vertices[vertex_idx] = np.fromstring(line, sep=" ", dtype=np.float64)[:3]

    triangles = []
    for _ in range(face_count):
        line, index = next_data_line(lines, index)
        values = np.fromstring(line, sep=" ", dtype=np.int64)
        if values.size < 4:
            continue
        polygon_size = int(values[0])
        polygon = values[1 : polygon_size + 1]
        if polygon_size < 3:
            continue
        for vertex_idx in range(1, polygon_size - 1):
            triangles.append((polygon[0], polygon[vertex_idx], polygon[vertex_idx + 1]))

    if not triangles:
        raise ValueError(f"{path} has no valid faces.")

    return vertices, np.asarray(triangles, dtype=np.int64)


def sample_pointcloud(vertices, faces, num_points, rng):
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)

    if np.allclose(areas.sum(), 0.0):
        choices = rng.integers(0, len(vertices), size=num_points)
        points = vertices[choices].copy()
    else:
        probabilities = areas / areas.sum()
        face_choices = rng.choice(len(triangles), size=num_points, p=probabilities)
        sampled_triangles = triangles[face_choices]

        u = rng.random(num_points)
        v = rng.random(num_points)
        sqrt_u = np.sqrt(u)

        points = (
            (1.0 - sqrt_u)[:, None] * sampled_triangles[:, 0]
            + (sqrt_u * (1.0 - v))[:, None] * sampled_triangles[:, 1]
            + (sqrt_u * v)[:, None] * sampled_triangles[:, 2]
        )

    points = points.astype(np.float32)
    points -= points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points, axis=1).max()
    if radius > 0:
        points /= radius
    rng.shuffle(points)
    return points


def collect_samples(source_root):
    class_names = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    splits = {"train": [], "test": []}

    for class_name in class_names:
        for split_name in ("train", "test"):
            split_dir = source_root / class_name / split_name
            for file_path in sorted(split_dir.glob("*.off")):
                splits[split_name].append((class_to_idx[class_name], class_name, file_path))

    return class_names, splits


def write_h5(output_path, split_name, split_samples, num_points, seed):
    total = len(split_samples)
    data = np.empty((total, num_points, 3), dtype=np.float32)
    label = np.empty((total, 1), dtype=np.int64)
    shape_ids = []
    shape_names = []

    print(f"[{split_name}] 开始转换，共 {total} 个样本 -> {output_path}")
    for index, (class_idx, class_name, file_path) in enumerate(split_samples, start=1):
        rng = np.random.default_rng(seed + index)
        vertices, faces = load_off_mesh(file_path)
        data[index - 1] = sample_pointcloud(vertices, faces, num_points, rng)
        label[index - 1, 0] = class_idx
        shape_ids.append(file_path.stem)
        shape_names.append(class_name)

        if index == total or index % 200 == 0:
            print(f"[{split_name}] 已完成 {index}/{total}")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.create_dataset("label", data=label)
        handle.create_dataset("shape_id", data=np.asarray(shape_ids, dtype=object), dtype=string_dtype)
        handle.create_dataset("shape_name", data=np.asarray(shape_names, dtype=object), dtype=string_dtype)


def write_metadata(output_root, class_names, num_points, split_counts):
    (output_root / "shape_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    metadata = {
        "num_points": num_points,
        "num_classes": len(class_names),
        "class_names": class_names,
        "split_counts": split_counts,
        "files": {
            "train": "ply_data_train0.h5",
            "test": "ply_data_test0.h5",
        },
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_root / "train_files.txt").write_text("ply_data_train0.h5\n", encoding="utf-8")
    (output_root / "test_files.txt").write_text("ply_data_test0.h5\n", encoding="utf-8")


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"未找到源目录：{source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    class_names, splits = collect_samples(source_root)

    split_counts = {split_name: len(samples) for split_name, samples in splits.items()}
    print(f"类别数={len(class_names)} 训练样本数={split_counts['train']} 测试样本数={split_counts['test']}")
    print(f"每个样本点数={args.num_points}")

    write_h5(output_root / "ply_data_train0.h5", "train", splits["train"], args.num_points, args.seed)
    write_h5(output_root / "ply_data_test0.h5", "test", splits["test"], args.num_points, args.seed + 100000)
    write_metadata(output_root, class_names, args.num_points, split_counts)

    print("转换完成")


if __name__ == "__main__":
    main()
