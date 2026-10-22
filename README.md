深度学习YOLOv8模型训练 火焰与烟雾分割数据集 并通过训练的权重建立深度学习YOLOV8或者烟雾分割检测系统 推理识别检测火焰和烟雾

以下代码及文字仅供参考学习使用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/042efef0b96941debb4d5b2b5da82179.png)
@[toc]

**数据集描述：**
火焰与烟雾分割数据集
yolov8格式数据集
 7110 张图像。
火灾烟雾数据以 YOLOv8 格式进行标注。


以下是根据您提供的“火焰与烟雾分割数据集”描述整理的结构化表格：

---

### **火焰与烟雾分割数据集信息表**

| 项目 | 内容 |
|------|------|
| **数据集名称** | 火焰与烟雾分割数据集 |
| **YOLO格式版本** | YOLOv8 格式标注 |
| **原始图像数量** | 7,110 张 |
| **增强后图像总数** | 约 355,500 张（每张原始图像生成 50 个增强版本） |
| **图像尺寸** | 统一调整为 640×640 像素（拉伸处理） |
| **预处理操作** | - 像素数据自动定向（去除 EXIF 定向信息）<br> - 图像尺寸统一为 640×640 |
| **数据增强策略** | 对每张原始图像生成 50 个不同增强版本，采用以下随机操作：<br> • **水平翻转**：概率为 50%<br> • **随机裁剪**：裁剪图像的 0% 至 15%<br> • **随机旋转**：角度范围为 -12° 至 +12°<br> • **随机剪切（Shearing）**：水平和垂直方向各 -3° 至 +3°<br> • **亮度调整**：随机变化 -20% 至 +20%<br> • **曝光度调整**：随机变化 -12% 至 +12%<br> • **高斯模糊**：模糊核大小为 0 至 0.5 像素<br> • **椒盐噪声**：对 1% 的像素点进行干扰 |
| **标注类型** | 实例分割（Segmentation） |
| **标注格式** | YOLOv8 分割格式（每个目标包含归一化的多边形轮廓点坐标） |
| **适用任务** | 火灾检测、火焰识别、烟雾识别、智能监控、应急响应系统等 |
| **数据特点** | - 高度增强，提升模型鲁棒性<br> - 包含多种光照、角度、遮挡与噪声模拟<br> - 适用于小样本扩展与复杂场景建模 |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fecfa94fdd894ea18ebf4b4c0e036294.png)

# **YOLOv8 火灾烟雾检测如何构建**

从环境搭建、数据集准备到模型训练、推理与评估的完整流程， **7110 张图像的火灾烟雾分割数据集（YOLOv8 格式）**。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a179456a7e5245dbb366eef4a16759ee.png)

## ✅ 一、环境搭建（CUDA + Anaconda + Python 虚拟环境）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d628f7d035854307ab45039f63aa3b17.png)

### 1. 安装 NVIDIA 驱动和 CUDA

确保 GPU 支持 CUDA，并安装最新版驱动：

```bash
# 查看显卡驱动是否正常
nvidia-smi
```

输出应显示 GPU 型号和驱动版本（如 535.xx 或更高）。  
然后根据驱动版本选择兼容的 CUDA Toolkit（推荐 CUDA 11.8 或 12.1）。

> 官方下载地址：[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

---

### 2. 安装 Anaconda

下载并安装 Anaconda（或 Miniconda）：
- 官网：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

安装完成后打开终端（Windows：Anaconda Prompt；Linux/macOS：Terminal）

---

### 3. 创建 Python 虚拟环境

```bash
# 创建名为 yolov8_env 的虚拟环境，使用 Python 3.10（推荐）
conda create -n yolov8_env python=3.10

# 激活环境
conda activate yolov8_env
```

---

### 4. 安装 PyTorch（支持 CUDA）

访问 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 获取适合你系统的命令。

例如（CUDA 11.8）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证安装：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应输出 True
```

---

### 5. 安装 YOLOv8 及其他依赖

```bash
# 安装 Ultralytics YOLOv8 官方库
pip install ultralytics

# 安装 OpenCV（图像处理）
pip install opencv-python opencv-python-headless

# 安装其他常用工具
pip install matplotlib numpy pandas tqdm tensorboard
```

验证 YOLO 安装：

```bash
yolo version
```

---

## ✅ 二、数据集结构说明（YOLOv8 格式）

数据集已经按 YOLOv8 格式预处理完毕，结构如下：

```
fire_smoke_dataset/
│
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   ├── val/
│   └── test/
│
└── data.yaml
```

### `data.yaml` 内容示例：

```yaml
# data.yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 2  # 类别数量
names: ['fire', 'smoke']  # 类别名称
```

> 💡 注意：每张图片对应一个 `.txt` 文件，内容为归一化后的 `(class_id, x_center, y_center, width, height)`，即 YOLO 格式。

---

## ✅ 三、使用官方预训练模型进行训练

### 1. 下载官方 YOLOv8 模型（自动）

Ultralytics 提供多个预训练模型：

| 模型 | 命令 |
|------|------|
| YOLOv8n (nano) | `yolov8n.pt` |
| YOLOv8s (small) | `yolov8s.pt` |
| YOLOv8m (medium) | `yolov8m.pt` |
| YOLOv8l (large) | `yolov8l.pt` |
| YOLOv8x (extra large) | `yolov8x.pt` |

训练时会自动下载（首次），也可手动下载：
- 地址：[https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases)

---

### 2. 训练代码（Python 脚本）

#### `train.py`

```python
from ultralytics import YOLO
import os

# 设置路径
DATA_YAML = 'fire_smoke_dataset/data.yaml'
MODEL_NAME = 'yolov8s.pt'  # 使用 small 模型
PROJECT = 'runs/fire_smoke'
NAME = 'yolov8s_fire_smoke'
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640

# 加载预训练模型
model = YOLO(MODEL_NAME)

# 开始训练
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    device=0 if torch.cuda.is_available() else 'cpu',  # 使用 GPU
    workers=4,
    optimizer='AdamW',           # 推荐用于小数据集
    lr0=0.001,                   # 初始学习率
    lrf=0.1,                     # 最终学习率 = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    patience=10,                 # 早停
    save=True,
    save_period=10,              # 每10轮保存一次
    cache=False,                 # 若内存充足可设为 True 加速
    augment=True                 # 数据增强（默认已开启）
)

print("训练完成！")
```

> 📌 训练日志和权重保存在 `runs/fire_smoke/yolov8s_fire_smoke/` 目录下。

---

## ✅ 四、推理代码（Inference）

### 1. 使用训练好的模型进行预测

#### `predict.py`

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/fire_smoke/yolov8s_fire_smoke/weights/best.pt')  # 或 last.pt

# 推理单张图片
def predict_image(img_path):
    results = model(img_path)
    result = results[0]
    
    # 显示结果
    annotated_img = result.plot()  # 绘制边界框和标签
    cv2.imshow('Prediction', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 打印检测信息
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = box.conf.item()
        label = model.names[cls_id]
        print(f"检测到: {label}, 置信度: {conf:.3f}")

# 示例调用
predict_image('test_image.jpg')
```

---

### 2. 实时摄像头推理

```python
def predict_camera():
    model = YOLO('runs/fire_smoke/yolov8s_fire_smoke/weights/best.pt')
    cap = cv2.VideoCapture(0)  # 默认摄像头

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640)
        annotated_frame = results[0].plot()

        cv2.imshow('YOLOv8 Fire & Smoke Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

predict_camera()
```

---

### 3. 视频文件或 RTSP 流推理

```python
def predict_video(video_path):
    model = YOLO('runs/fire_smoke/yolov8s_fire_smoke/weights/best.pt')
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640)
        annotated_frame = results[0].plot()

        cv2.imshow('Video Inference', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 示例：本地视频或 RTSP
predict_video('rtsp://admin:password@192.168.1.100:554/stream1')
```

---

## ✅ 五、评估代码（Validation / Test）

### 1. 在验证集上评估模型性能

```python
# eval.py
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/fire_smoke/yolov8s_fire_smoke/weights/best.pt')

# 在 val 或 test 集上评估
metrics = model.val(
    data='fire_smoke_dataset/data.yaml',
    split='val',  # 或 'test'
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',
    save_json=True,  # 生成 COCO 格式的 results.json（可用于 mAP 计算）
    conf=0.25,
    iou=0.6
)

# 输出指标
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.precision.mean():.4f}")
print(f"Recall: {metrics.box.recall.mean():.4f}")
```

---

### 2. 可视化 PR 曲线、混淆矩阵等

训练完成后，Ultralytics 会在 `runs/train/yolov8s_fire_smoke/` 自动生成以下图表：

- `results.png`：训练损失、mAP、Precision、Recall 曲线
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`

也可手动调用：

```python
model.val(
    plots=True,
    save_json=True
)
```

---

## ✅ 六、高级技巧与建议

### 1. 自定义数据增强（数据集中增强）

对原始图像做了丰富的增强（翻转、裁剪、旋转、亮度/曝光调整、模糊、噪声），这非常有助于提升泛化能力。

YOLOv8 默认也启用 Mosaic、MixUp 等增强，可在训练时关闭以避免过增强：

```python
model.train(..., mosaic=0.0, mixup=0.0)
```

---

### 2. 使用 TensorBoard 查看训练过程

```bash
tensorboard --logdir=runs/fire_smoke/yolov8s_fire_smoke
```

浏览器访问 `http://localhost:6006` 查看动态曲线。

---

### 3. 导出模型为 ONNX / TensorRT / CoreML 等格式

```python
model = YOLO('runs/fire_smoke/yolov8s_fire_smoke/weights/best.pt')
model.export(format='onnx', imgsz=640)      # ONNX
model.export(format='engine', imgsz=640)    # TensorRT (需安装 tensorrt)
model.export(format='coreml')               # iOS
model.export(format='tflite')               # Android
```

---

以上文字及代码仅供参考学习使用
