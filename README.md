# 基于 YOLOv8 的工业瓶体缺陷检测系统

## 1. 项目简介

本项目基于 MVTec AD 数据集中的 `bottle` 类别，构建了一个工业瓶体缺陷检测系统。项目首先分析原始数据集中图像、缺陷类别与像素级标注之间的对应关系，然后编写数据转换脚本，将 MVTec AD 提供的缺陷 mask 转换为 YOLO 目标检测格式的边界框标签，最终使用 YOLOv8n 完成瓶体缺陷检测模型的训练、验证与推理。

系统能够对瓶体图像中的缺陷进行自动定位与分类，检测类别包括大破损、小破损和污染缺陷，可用于模拟工业视觉质检中的 OK / NG 判断流程。

## 2. 项目任务

- 任务类型：工业缺陷检测 / 目标检测
- 数据集：MVTec AD - bottle
- 检测模型：YOLOv8n
- 输入：瓶体图像
- 输出：缺陷类别、置信度、检测框位置
- 检测类别：
  - `broken_large`：大破损
  - `broken_small`：小破损
  - `contamination`：污染缺陷

## 3. 原始数据集结构

原始 MVTec AD bottle 数据集结构如下：

```text
bottle
├── ground_truth
├── test
├── train
├── license.txt
└── readme.txt
```

其中各部分含义如下：

| 路径 | 含义 |
|---|---|
| `train/good` | 正常瓶体图像，原始训练集 |
| `test/good` | 正常瓶体测试图像 |
| `test/broken_large` | 大破损缺陷图像 |
| `test/broken_small` | 小破损缺陷图像 |
| `test/contamination` | 污染缺陷图像 |
| `ground_truth/broken_large` | 大破损缺陷对应的 mask 标注 |
| `ground_truth/broken_small` | 小破损缺陷对应的 mask 标注 |
| `ground_truth/contamination` | 污染缺陷对应的 mask 标注 |

MVTec AD 原始数据集主要面向异常检测和缺陷分割任务，其缺陷标注形式为像素级 mask，而 YOLOv8 目标检测任务需要的是边界框标签。因此，本项目首先进行了数据格式转换。

## 4. 数据处理方法

本项目将 MVTec AD 的 mask 标注转换为 YOLO 检测框标签，主要步骤如下：

1. 读取原始缺陷图像及其对应的 `ground_truth` mask；
2. 对 mask 图像进行灰度读取和二值化处理；
3. 提取白色缺陷区域的外部轮廓；
4. 根据轮廓计算最小外接矩形；
5. 将矩形框转换为 YOLO 格式标签；
6. 将正常样本生成空标签文件；
7. 重新划分训练集、验证集和测试集；
8. 自动生成 `data.yaml` 配置文件。

YOLO 标签格式如下：

```text
class_id x_center y_center width height
```

其中 `x_center`、`y_center`、`width` 和 `height` 均为归一化后的数值，取值范围为 0 到 1。

## 5. 转换后的 YOLO 数据集结构

转换后生成的数据集目录为：

```text
bottle_yolo
├── images
│   ├── train
│   ├── val
│   └── test
├── labels
│   ├── train
│   ├── val
│   └── test
├── data.yaml
└── summary.txt
```

数据集划分情况如下：

| 数据集 | 总图片数 | 缺陷图片数 | 正常图片数 |
|---|---:|---:|---:|
| train | 204 | 42 | 162 |
| val | 58 | 12 | 46 |
| test | 30 | 9 | 21 |

`data.yaml` 配置如下：

```yaml
path: C:/D/pyproject/bottle/bottle_yolo
train: images/train
val: images/val
test: images/test

names:
  0: broken_large
  1: broken_small
  2: contamination
```

## 6. 环境配置

本项目主要依赖如下：

- Python 3.10
- PyTorch 2.11.0 + CUDA 12.8
- Ultralytics 8.4.57
- OpenCV
- PyYAML
- tqdm

安装依赖：

```bash
pip install ultralytics opencv-python pyyaml tqdm
```

## 7. 数据集转换

在项目目录下执行：

```bash
cd /d C:\D\pyproject\bottle
python prepare_bottle_yolo.py --src ".\bottle" --out ".\bottle_yolo" --mode multiclass
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--src` | 原始 MVTec AD bottle 数据集路径 |
| `--out` | 转换后的 YOLO 数据集输出路径 |
| `--mode multiclass` | 使用三分类缺陷检测模式 |

## 8. 模型训练

本项目使用 YOLOv8n 作为基础检测模型。训练命令如下：

```bash
yolo detect train model=yolov8n.pt data="bottle_yolo\data.yaml" epochs=100 imgsz=640 batch=16 device=0
```

主要训练参数如下：

| 参数 | 数值 |
|---|---|
| 模型 | YOLOv8n |
| 输入尺寸 | 640 × 640 |
| 训练轮数 | 100 |
| batch size | 16 |
| optimizer | auto |
| 初始学习率 | 0.01 |
| weight decay | 0.0005 |
| 数据增强 | mosaic、flip、scale、hsv 等 |
| 训练设备 | NVIDIA GeForce RTX 5060 |

训练完成后，模型权重保存于：

```text
runs/detect/train/weights/best.pt
```

## 9. 模型验证

验证命令如下：

```bash
yolo detect val model="runs\detect\train\weights\best.pt" data="bottle_yolo\data.yaml" split=val device=0
```

当前验证集整体结果如下：

| 指标 | 数值 |
|---|---:|
| Precision | 0.748 |
| Recall | 0.714 |
| mAP50 | 0.721 |
| mAP50-95 | 0.493 |

各类别结果如下：

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 58 | 13 | 0.748 | 0.714 | 0.721 | 0.493 |
| broken_large | 3 | 3 | 0.696 | 1.000 | 0.830 | 0.529 |
| broken_small | 4 | 5 | 0.786 | 0.742 | 0.755 | 0.450 |
| contamination | 5 | 5 | 0.764 | 0.400 | 0.577 | 0.498 |

说明：

- `Precision` 表示模型检测出的缺陷中有多少是真实缺陷；
- `Recall` 表示真实缺陷中有多少被模型成功检测出；
- `mAP50` 表示 IoU 阈值为 0.5 时的平均检测精度；
- `mAP50-95` 表示 IoU 阈值从 0.5 到 0.95 时的综合平均检测精度。

## 10. 推理速度

验证阶段输出的速度信息如下：

```text
Speed: 0.3ms preprocess, 1.5ms inference, 0.5ms postprocess per image
```

因此，单张图像完整处理时间约为：

```text
0.3 + 1.5 + 0.5 = 2.3 ms/image
```

完整流程 FPS 约为：

```text
FPS = 1000 / 2.3 ≈ 434.8 FPS
```

如果只统计模型前向推理阶段，则：

```text
Inference FPS = 1000 / 1.5 ≈ 666.7 FPS
```

## 11. 模型预测

对转换后的 YOLO 测试集进行预测：

```bash
yolo detect predict model="runs\detect\train\weights\best.pt" source="bottle_yolo\images\test" conf=0.25 save=True device=0
```

也可以对原始 MVTec AD bottle 中的某一类缺陷图像进行预测：

```bash
yolo detect predict model="runs\detect\train\weights\best.pt" source="bottle\test\broken_large" conf=0.25 save=True device=0
```

预测结果默认保存到：

```text
runs/detect/predict
```

## 12. 项目流程

整体流程如下：

```text
MVTec AD bottle 数据集
        ↓
读取原始图像与 ground_truth mask
        ↓
mask 二值化与轮廓提取
        ↓
转换为 YOLO bbox 标签
        ↓
划分 train / val / test
        ↓
配置 data.yaml
        ↓
训练 YOLOv8n 缺陷检测模型
        ↓
验证 Precision、Recall、mAP 等指标
        ↓
预测瓶体图像并输出缺陷框
        ↓
实现工业质检 OK / NG 判断
```

## 13. 项目亮点

1. **完成非 YOLO 数据集到 YOLO 检测数据集的转换**  
   针对 MVTec AD 原始像素级 mask 标注，编写脚本完成 mask 到 bounding box 的自动转换。

2. **构建工业瓶体缺陷检测数据集**  
   将原始 bottle 数据重新整理为 YOLOv8 所需的 `images/labels` 结构，并完成 train、val、test 划分。

3. **实现多类别缺陷定位与识别**  
   模型能够识别 `broken_large`、`broken_small` 和 `contamination` 三类瓶体缺陷。

4. **完成训练、验证和预测闭环**  
   项目包含数据预处理、模型训练、指标评估、缺陷预测和结果可视化完整流程。

5. **具备工业质检系统扩展能力**  
   可进一步封装为 OK / NG 检测脚本或图形化界面，用于模拟工业产线视觉检测流程。

## 14. 后续改进方向

1. **扩大数据规模**  
   当前仅使用 MVTec AD bottle 类别，可进一步扩展到 capsule、hazelnut、metal_nut 等更多工业类别。

2. **改进小缺陷检测能力**  
   针对 `broken_small` 等小目标缺陷，可以尝试提高输入分辨率、使用更大的 YOLO 模型或加入注意力机制。

3. **提升污染类缺陷召回率**  
   当前 `contamination` 类别 Recall 相对较低，后续可通过数据增强、类别重采样或损失函数调整改善。

4. **增加工业检测 demo**  
   封装输入图像、模型推理、OK / NG 判断、检测日志保存等功能，使项目更接近真实工业应用。

5. **进行多模型对比实验**  
   可进一步比较 YOLOv8n、YOLOv8s、YOLOv5、YOLOv11 等模型在精度和速度上的差异。

## 15. 项目总结

本项目围绕工业瓶体缺陷检测任务，完成了从数据集分析、标注格式转换、YOLO 数据集构建到模型训练、验证和推理的完整流程。通过将 MVTec AD bottle 数据集中的像素级缺陷 mask 转换为 YOLO 边界框标签，成功构建了适用于目标检测任务的工业缺陷数据集，并基于 YOLOv8n 实现了对瓶体大破损、小破损和污染缺陷的自动定位与分类。实验结果表明，该方法能够较好地完成瓶体缺陷检测任务，并具备进一步扩展为工业质检系统的基础。
