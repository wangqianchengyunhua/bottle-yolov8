import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


DEFECT_CLASSES = {
    "broken_large": 0,
    "broken_small": 1,
    "contamination": 2,
}

def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_bbox_from_mask(mask_path: Path, img_w: int, img_h: int):
    """
    把 MVTec 的 mask 转成 YOLO bbox
    返回格式: x_center, y_center, width, height，均为 0~1
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w <= 1 or h <= 1:
            continue

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        bw = w / img_w
        bh = h / img_h

        boxes.append((x_center, y_center, bw, bh))

    return boxes


def collect_samples(src: Path, mode: str):
    """
    收集 bottle 样本。
    缺陷样本：test/broken_large 等 + ground_truth mask
    正常样本：train/good + test/good，标签为空
    """
    samples = []

    # 缺陷样本
    for defect_name, cls_id in DEFECT_CLASSES.items():
        img_dir = src / "test" / defect_name
        mask_dir = src / "ground_truth" / defect_name

        if not img_dir.exists():
            print(f"[WARN] 缺陷图片目录不存在: {img_dir}")
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            mask_name = img_path.stem + "_mask.png"
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                print(f"[WARN] 找不到对应 mask: {mask_path}")
                continue

            if mode == "binary":
                label_cls = 0
            else:
                label_cls = cls_id

            samples.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "label_cls": label_cls,
                "is_defect": True,
            })

    # 正常样本，YOLO 标签为空
    normal_dirs = [
        src / "train" / "good",
        src / "test" / "good",
    ]

    for normal_dir in normal_dirs:
        if not normal_dir.exists():
            continue

        for img_path in sorted(normal_dir.glob("*.png")):
            samples.append({
                "img_path": img_path,
                "mask_path": None,
                "label_cls": None,
                "is_defect": False,
            })

    return samples


def split_samples(samples, train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def save_yolo_dataset(splits, out: Path, mode: str):
    for split in ["train", "val", "test"]:
        mkdir(out / "images" / split)
        mkdir(out / "labels" / split)

    for split, samples in splits.items():
        print(f"\n处理 {split}: {len(samples)} 张图片")

        for idx, sample in enumerate(tqdm(samples)):
            img_path = sample["img_path"]

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] 无法读取图片: {img_path}")
                continue

            img_h, img_w = img.shape[:2]

            # 为了避免不同文件夹里 000.png 重名，这里重新命名
            new_name = f"{split}_{idx:05d}.png"

            dst_img_path = out / "images" / split / new_name
            dst_label_path = out / "labels" / split / new_name.replace(".png", ".txt")

            shutil.copy2(img_path, dst_img_path)

            label_lines = []

            if sample["is_defect"]:
                boxes = get_bbox_from_mask(sample["mask_path"], img_w, img_h)

                for box in boxes:
                    x_center, y_center, bw, bh = box
                    cls_id = sample["label_cls"]
                    label_lines.append(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                    )

            # 正常图片生成空 txt
            with open(dst_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    if mode == "binary":
        names = {0: "defect"}
    else:
        names = {
            0: "broken_large",
            1: "broken_small",
            2: "contamination",
        }

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }

    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        for split, samples in splits.items():
            defect_count = sum(1 for s in samples if s["is_defect"])
            good_count = len(samples) - defect_count
            f.write(f"{split}: total={len(samples)}, defect={defect_count}, good={good_count}\n")

    print("\n转换完成！")
    print(f"YOLO 数据集输出目录: {out}")
    print(f"data.yaml 路径: {out / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="MVTec bottle 原始路径")
    parser.add_argument("--out", type=str, required=True, help="输出 YOLO 数据集路径")
    parser.add_argument(
        "--mode",
        type=str,
        default="multiclass",
        choices=["multiclass", "binary"],
        help="multiclass=三类缺陷；binary=只检测 defect 一类",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        raise FileNotFoundError(f"找不到原始 bottle 路径: {src}")

    samples = collect_samples(src, args.mode)

    if len(samples) == 0:
        raise RuntimeError("没有收集到任何样本，请检查路径是否正确。")

    print(f"共收集到 {len(samples)} 张图片")

    splits = split_samples(samples, seed=args.seed)
    save_yolo_dataset(splits, out, args.mode)


if __name__ == "__main__":
    main()