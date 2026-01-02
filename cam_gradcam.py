"""
cam_gradcam.py - CAM and Grad-CAM Implementation for CIFAR-10
用于CIFAR-10图像分类的可视化解释系统

实现了：
1. CAM (Class Activation Mapping)
2. Grad-CAM (Gradient-weighted Class Activation Mapping)
3. 可视化工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os


# ==================== CAM实现 ====================

class CAM:
    """
    Class Activation Mapping
    
    工作原理：
    1. 获取最后一层卷积特征图 [B, C, H, W]
    2. 获取FC层的权重 [num_classes, C]
    3. 对于目标类别c，CAM = Σ(w_c_k * feature_k)
    
    要求：模型必须有GAP + FC结构
    """
    def __init__(self, model):
        """
        model: 支持CAM的模型（必须有GAP + FC结构，且forward支持return_feature参数）
        """
        self.model = model
        self.model.eval()
        
    def generate_cam(self, input_image, target_class=None):
        """
        生成CAM热力图
        
        参数:
        - input_image: [1, C, H, W] 输入图像tensor
        - target_class: 目标类别，如果为None则使用预测类别
        
        返回:
        - cam: 归一化的CAM热力图 [H, W]，范围[0, 1]
        - predicted_class: 预测的类别索引
        - class_score: 目标类别的logit得分
        """
        self.model.eval()
        
        # 前向传播获取特征图和输出
        with torch.no_grad():
            output, features = self.model(input_image, return_feature=True)
            
        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        class_score = output[0, target_class].item()
        
        # 获取FC层权重 [num_classes, C]
        fc_weights = self.model.fc.weight.data  # [10, 512]
        
        # 获取目标类别的权重
        target_weights = fc_weights[target_class]  # [512]
        
        # 计算CAM
        features = features.squeeze(0)  # [512, H, W]
        
        # 使用einsum进行向量化计算，比循环更快且更稳定
        # 'k,khw->hw': 对通道维度k进行加权求和
        cam = torch.einsum('k,khw->hw', target_weights, features)
        
        # 打印调试信息（如果全为0，说明计算有问题）
        if cam.max() == cam.min():
            print(f"Warning: CAM is constant! min={cam.min()}, max={cam.max()}")
            print(f"Features min={features.min()}, max={features.max()}")
            print(f"Weights min={target_weights.min()}, max={target_weights.max()}")
        
        # ReLU和归一化
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        
        # 避免除零
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        return cam, target_class, class_score


# ==================== Grad-CAM实现 ====================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    
    工作原理：
    1. 前向传播到目标层，保存特征图 A
    2. 继续前向传播得到类别得分 y^c
    3. 反向传播计算 dy^c/dA
    4. 对梯度进行全局平均池化得到权重 α_k = (1/Z) Σ Σ (dy^c/dA_k)
    5. Grad-CAM = ReLU(Σ(α_k * A_k))
    
    优势：适用于任何CNN架构
    """
    def __init__(self, model, target_layer):
        """
        参数:
        - model: 任何CNN模型
        - target_layer: 目标层（nn.Module），通常是最后一个卷积层
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # 注册hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向和反向hooks来捕获特征图和梯度"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        """
        生成Grad-CAM热力图
        
        参数:
        - input_image: [1, C, H, W] 输入图像tensor
        - target_class: 目标类别，如果为None则使用预测类别
        
        返回:
        - cam: 归一化的Grad-CAM热力图 [H, W]，范围[0, 1]
        - predicted_class: 预测的类别索引
        - class_score: 目标类别的logit得分
        """
        self.model.eval()
        input_image = input_image.clone()
        input_image.requires_grad = True
        
        # 前向传播
        output = self.model(input_image)
        
        # 确定目标类别
        predicted_class = output.argmax(dim=1).item()
        if target_class is None:
            target_class = predicted_class
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # 计算权重（对梯度进行全局平均池化）
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # α_k = (1/Z) * Σ Σ (dy^c/dA_k)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Grad-CAM = ReLU(Σ(α_k * A_k))
        cam = torch.sum(weights * activations, dim=1).squeeze(0)  # [H, W]
        cam = torch.relu(cam)
        
        # 归一化
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        return cam, target_class, class_score.item()


# ==================== 可视化工具 ====================

def visualize_cam(original_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    将CAM热力图叠加到原始图像上
    
    参数:
    - original_image: PIL Image或numpy array (H, W, C)，RGB格式，范围[0, 255]
    - cam: CAM热力图 [H, W]，范围[0, 1]
    - alpha: 热力图透明度
    - colormap: OpenCV颜色映射
    
    返回: 
    - overlayed: 叠加后的图像 (numpy array, RGB)
    - cam_colored: 纯热力图 (numpy array, RGB)
    """
    # 转换原始图像
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # 调整CAM大小以匹配原始图像（使用双三次插值并高斯平滑以改善视觉效果）
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = cv2.GaussianBlur(cam_resized, (7, 7), 0)

    # 百分位对比度拉伸 + 伽马增强，让低值也能更易被看见
    # 将 [p5, p95] 映射到 [0, 1]，并做 gamma（<1 提升亮部）
    p_low, p_high = np.percentile(cam_resized, [5, 95])
    if p_high > p_low:
        cam_resized = (cam_resized - p_low) / (p_high - p_low + 1e-8)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)
    # gamma 调整：0.8 可稍微增加对比度，可按需调小（如 0.6）
    cam_resized = np.power(cam_resized, 0.8)

    # 应用颜色映射
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlayed = cam_colored * alpha + original_image * (1 - alpha)
    overlayed = np.uint8(overlayed)
    
    return overlayed, cam_colored


def plot_comparison(original_img, cam_heatmap, gradcam_heatmap, 
                   predicted_class, true_class, class_names, 
                   cam_score, gradcam_score):
    """
    绘制原始图像、CAM和Grad-CAM的对比图
    
    参数:
    - original_img: 原始图像 numpy array
    - cam_heatmap: CAM热力图
    - gradcam_heatmap: Grad-CAM热力图
    - predicted_class: 预测类别索引
    - true_class: 真实类别索引
    - class_names: 类别名称列表
    - cam_score: CAM目标类别得分
    - gradcam_score: Grad-CAM目标类别得分
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：叠加图
    # 原始图像
    axes[0, 0].imshow(original_img, interpolation='bilinear')
    axes[0, 0].set_title(f'Original Image\nTrue: {class_names[true_class]}\nPred: {class_names[predicted_class]}', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # CAM叠加
    cam_overlay, _ = visualize_cam(original_img, cam_heatmap)
    axes[0, 1].imshow(cam_overlay, interpolation='bilinear')
    axes[0, 1].set_title(f'CAM Overlay\nScore: {cam_score:.3f}', fontsize=12)
    axes[0, 1].axis('off')
    
    # Grad-CAM叠加
    gradcam_overlay, _ = visualize_cam(original_img, gradcam_heatmap)
    axes[0, 2].imshow(gradcam_overlay, interpolation='bilinear')
    axes[0, 2].set_title(f'Grad-CAM Overlay\nScore: {gradcam_score:.3f}', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行：纯热力图
    axes[1, 0].axis('off')
    
    im1 = axes[1, 1].imshow(cam_heatmap, cmap='jet', interpolation='bilinear')
    axes[1, 1].set_title('CAM Heatmap', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    
    im2 = axes[1, 2].imshow(gradcam_heatmap, cmap='jet', interpolation='bilinear')
    axes[1, 2].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    return fig


# ==================== 评估和演示 ====================

def test_and_visualize(model, test_loader, cam_generator, gradcam_generator, 
                       device, num_samples=10, save_dir='./results'):
    """
    测试模型并可视化CAM和Grad-CAM
    
    选择正确和错误预测的样本进行可视化
    
    参数:
    - model: 模型
    - test_loader: 测试数据加载器
    - cam_generator: CAM生成器
    - gradcam_generator: Grad-CAM生成器
    - device: 计算设备
    - num_samples: 总样本数
    - save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    model.eval()
    
    correct_samples = []
    incorrect_samples = []
    
    # 收集样本
    print("收集测试样本...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(images)):
                pred = predicted[i].item()
                true = labels[i].item()
                
                if pred == true and len(correct_samples) < num_samples // 2:
                    correct_samples.append((images[i:i+1], true, pred))
                elif pred != true and len(incorrect_samples) < num_samples // 2:
                    incorrect_samples.append((images[i:i+1], true, pred))
                
                if len(correct_samples) >= num_samples // 2 and len(incorrect_samples) >= num_samples // 2:
                    break
            
            if len(correct_samples) >= num_samples // 2 and len(incorrect_samples) >= num_samples // 2:
                break
    
    print(f"收集到 {len(correct_samples)} 个正确预测样本，{len(incorrect_samples)} 个错误预测样本")
    
    # 可视化
    all_samples = correct_samples + incorrect_samples
    
    for idx, (img_tensor, true_label, pred_label) in enumerate(all_samples):
        print(f"生成可视化 {idx+1}/{len(all_samples)}...")
        
        # 反归一化：从[-1, 1]到[0, 255]
        img_denorm = img_tensor.clone().squeeze()
        img_denorm = img_denorm * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        img_denorm = img_denorm.permute(1, 2, 0).cpu().numpy()
        img_denorm = np.clip(img_denorm * 255, 0, 255).astype(np.uint8)
        
        # 生成CAM
        cam_heatmap, _, cam_score = cam_generator.generate_cam(img_tensor)
        
        # 生成Grad-CAM
        gradcam_heatmap, _, gradcam_score = gradcam_generator.generate_cam(img_tensor)
        
        # 绘制对比图
        fig = plot_comparison(img_denorm, cam_heatmap, gradcam_heatmap,
                             pred_label, true_label, class_names,
                             cam_score, gradcam_score)
        
        # 保存
        status = 'correct' if pred_label == true_label else 'incorrect'
        filename = f'sample_{idx:02d}_{status}_{class_names[true_label]}_pred_{class_names[pred_label]}.png'
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"所有可视化结果已保存到: {save_dir}")


def evaluate_model_accuracy(model, test_loader, device):
    """
    评估模型在测试集上的准确率
    
    返回:
    - accuracy: 测试准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0
    
    print("评估模型准确率...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


# ==================== 主函数 ====================

def main():
    """
    主函数：加载模型，生成CAM和Grad-CAM可视化
    """
    import argparse
    parser = argparse.ArgumentParser(description='生成CAM和Grad-CAM可视化')
    parser.add_argument('--model', type=str, default='VGG_CAM',
                       choices=['VGG_CAM', 'ResNet_CAM'],
                       help='选择模型: VGG_CAM 或 ResNet_CAM')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='可视化样本数量')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据加载
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(root='./', train=False, download=False, 
                                    transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # 加载模型
    from models import VGG_CAM, ResNet_CAM
    
    print('\n' + '='*60)
    print(f'加载{args.model}模型...')
    print('='*60)
    
    if args.model == 'VGG_CAM':
        model = VGG_CAM().to(device)
        target_layer_name = 'layer4'
        target_layer = model.layer4[-2]  # 选择最后一个卷积层(Conv2d)
        save_dir = './results/task1_vgg'
    else:  # ResNet_CAM
        model = ResNet_CAM().to(device)
        # 使用更浅的 layer3 以获得更高的空间分辨率热力图（例如 8x8）
        target_layer_name = 'layer3'
        try:
            target_layer = model.layer3[-1].conv2
        except Exception:
            # 兼容不同 ResNet 实现
            target_layer = model.layer3[-1]
        save_dir = './results/task1_resnet'
    
    # 加载预训练权重
    checkpoint_path = f'./checkpoint/{args.model}/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f'从 {checkpoint_path} 加载权重...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型，训练轮次: {checkpoint['epoch']}, 准确率: {checkpoint['accuracy']:.2f}%")
    else:
        print(f"警告: 未找到预训练权重 {checkpoint_path}")
        print("将使用随机初始化的模型（结果可能不佳）")
        print(f"请先运行: python train_cam_model.py --model {args.model}")
    
    # 评估模型
    accuracy = evaluate_model_accuracy(model, test_loader, device)
    print(f'\n测试集准确率: {accuracy:.2f}%\n')
    
    # 初始化CAM和Grad-CAM
    print('初始化CAM和Grad-CAM...')
    cam_generator = CAM(model)
    
    print(f'目标层: {target_layer_name}')
    gradcam_generator = GradCAM(model, target_layer=target_layer)
    
    # 生成可视化
    print('\n生成可视化结果...')
    test_loader_single = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)
    test_and_visualize(model, test_loader_single, cam_generator, gradcam_generator, 
                      device, num_samples=args.num_samples, save_dir=save_dir)
    
    print('\n' + '='*60)
    print('完成！')
    print('='*60)


if __name__ == '__main__':
    main()
