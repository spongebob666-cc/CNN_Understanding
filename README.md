# CAM & Grad-CAM 可视化实现

基于CIFAR-10数据集的类激活映射(CAM)和梯度加权类激活映射(Grad-CAM)实现，用于深度学习模型的可视化解释。

## 📁 项目结构

```
cvlab6/
├── models.py                    # 网络模型定义 (VGG, ResNet, VGG_CAM, ResNet_CAM)
├── main.py                      # 标准模型训练脚本
├── train_cam_model.py          # CAM模型训练脚本
├── train_cam_model_gpu.py      # GPU加速训练脚本（可选）
├── cam_gradcam.py              # CAM/Grad-CAM核心实现
├── task2_evaluation.py         # 评估指标计算
├── task3_detection.py          # 目标检测应用
├── run_all_tasks.py            # 一键运行所有任务
├── checkpoint/                 # 训练好的模型权重
├── results/                    # 可视化结果输出
├── logs/                       # TensorBoard训练日志
└── cifar-10-batches-py/       # CIFAR-10数据集
```

## 🚀 快速开始

### 环境要求

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib opencv-python pillow tensorboard tqdm
```

**Python版本**: 3.7+  
**PyTorch版本**: 1.8+

### 一键运行所有任务

```bash
python run_all_tasks.py --model ResNet_CAM
```

这将依次完成：
1. ✅ 训练CAM模型（如果不存在）
2. ✅ 生成CAM和Grad-CAM可视化
3. ✅ 计算评估指标
4. ✅ 应用于目标检测（可选）

## 📚 详细使用指南

### Task 1: CAM与Grad-CAM可视化

#### 1.1 训练CAM模型

```bash
# 训练ResNet_CAM（推荐）
python train_cam_model.py --model ResNet_CAM --epochs 50 --batch_size 128

# 训练VGG_CAM
python train_cam_model.py --model VGG_CAM --epochs 50 --batch_size 128

# GPU加速训练（如有GPU）
python train_cam_model_gpu.py --model ResNet_CAM --epochs 50 --use_amp
```

**参数说明**：
- `--model`: 模型选择 (`ResNet_CAM`, `VGG_CAM`)
- `--epochs`: 训练轮数（默认50）
- `--batch_size`: 批次大小（默认128）
- `--lr`: 学习率（默认0.1）

#### 1.2 生成可视化结果

```bash
# 生成CAM和Grad-CAM对比图
python cam_gradcam.py --model ResNet_CAM --num_samples 10

# 指定保存路径
python cam_gradcam.py --model ResNet_CAM --output_dir ./results/my_results
```

**输出内容**：
- 原始图像、CAM热力图、Grad-CAM热力图
- 保存在 `./results/task1_{model_name}/` 目录

### Task 2: 评估指标计算

```bash
# 对比CAM和Grad-CAM性能
python task2_evaluation.py --model ResNet_CAM --num_samples 100
```

**评估指标**：
1. **Localization Accuracy**: 热力图定位准确性
2. **Interpretability Score**: 可解释性得分
3. **Faithfulness**: 模型预测一致性

**输出文件**：
- `./results/task2/metrics_summary.txt`: 统计摘要
- `./results/task2/comparison_plot.png`: 对比图表

### Task 3: 目标检测应用

```bash
# 在目标检测任务上应用Grad-CAM
python task3_detection.py --num_samples 10
```

**功能**：
- 使用预训练的Faster R-CNN模型
- 为检测框生成Grad-CAM热力图
- 可视化模型关注区域

**输出目录**：`./results/task3/`

## 📊 结果查看

### TensorBoard可视化

```bash
# 查看训练曲线
tensorboard --logdir ./logs
```

在浏览器打开 http://localhost:6006

### 结果文件结构

```
results/
├── task1_resnet_cam/        # Task 1可视化结果
│   ├── sample_0.png
│   ├── sample_1.png
│   └── ...
├── task2/                   # Task 2评估结果
│   ├── metrics_summary.txt
│   └── comparison_plot.png
└── task3/                   # Task 3检测结果
    └── detection_*.png
```

## 🔧 核心功能说明

### CAM (Class Activation Mapping)

- **原理**: 使用全局平均池化替换全连接层，通过特征图加权生成热力图
- **优点**: 无需反向传播，计算快速
- **限制**: 需要特定网络结构（GAP层）

**实现位置**: [cam_gradcam.py](cam_gradcam.py) - `generate_cam()` 函数

### Grad-CAM (Gradient-weighted CAM)

- **原理**: 使用梯度加权特征图，生成类别相关的热力图
- **优点**: 适用于任意CNN架构，不需要修改网络结构
- **应用**: 可用于任意层的可视化

**实现位置**: [cam_gradcam.py](cam_gradcam.py) - `generate_gradcam()` 函数

## 📝 模型说明

### 可用模型

| 模型 | 参数量 | CIFAR-10准确率 | 训练时间 |
|------|--------|----------------|----------|
| VGG_CAM | ~15M | ~87% | ~30分钟 |
| ResNet_CAM | ~11M | ~90% | ~20分钟 |

所有模型都已预训练并保存在 `./checkpoint/` 目录。

### 模型结构特点

- **VGG_CAM**: 在最后的卷积层后添加GAP层，用于CAM生成
- **ResNet_CAM**: 在残差块后添加GAP层，适合深层网络

## 🎯 常见问题

### Q1: 训练好的模型在哪里？
A: 保存在 `./checkpoint/{模型名}/best_model.pth`

### Q2: 如何只生成Grad-CAM不重新训练？
A: 直接运行 `python cam_gradcam.py --model ResNet_CAM`，会自动加载已有模型

### Q3: 如何修改可视化的样本数量？
A: 使用 `--num_samples` 参数，例如 `--num_samples 20`

### Q4: GPU内存不足怎么办？
A: 减小batch_size，例如 `--batch_size 64` 或 `--batch_size 32`

### Q5: 如何使用自己的图像？
A: 修改 `cam_gradcam.py` 中的数据加载部分，加载自定义图像

## 📖 参考资料

- **CAM论文**: Zhou et al. "Learning Deep Features for Discriminative Localization" (CVPR 2016)
- **Grad-CAM论文**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)

## 📄 License

本项目仅用于学术研究和教学目的。

## 👨‍💻 作者

计算机视觉实验 Lab 6 - CAM & Grad-CAM Implementation

---

**最后更新**: 2026年1月2日
