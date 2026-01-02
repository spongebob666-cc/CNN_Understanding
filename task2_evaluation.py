"""
task2_evaluation.py - Task 2: CAM/Grad-CAM评估指标

实现了：
1. Deletion Metric: 按重要性逐步删除像素
2. Insertion Metric: 按重要性逐步插入像素
3. Average Drop: 删除重要区域后的平均分数下降
4. 批量评估和可视化
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import os
from tqdm import tqdm


class EvaluationMetrics:
    """
    CAM/Grad-CAM评估指标类
    
    用于定量评估视觉解释方法的质量
    """
    
    def __init__(self, model, device):
        """
        参数:
        - model: 待评估的CNN模型
        - device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def deletion_metric(self, image, heatmap, target_class, num_steps=50, blur_sigma=10):
        """
        Deletion指标
        
        原理：
        逐步删除（模糊）heatmap中最重要的像素，观察目标类得分下降速度
        好的解释应该导致得分快速下降
        
        参数:
        - image: [1, C, H, W] 原始图像tensor
        - heatmap: [H, W] 热力图，范围[0, 1]
        - target_class: 目标类别索引
        - num_steps: 删除步骤数
        - blur_sigma: 高斯模糊的sigma值
        
        返回:
        - deletion_auc: AUC值（曲线下面积），越小越好
        - score_curve: 每一步的得分列表
        """
        self.model.eval()
        
        # 调整heatmap大小到图像尺寸
        img_h, img_w = image.shape[2], image.shape[3]
        heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
        
        # 按重要性排序像素位置
        flat_heatmap = heatmap_resized.flatten()
        sorted_indices = np.argsort(flat_heatmap)[::-1]  # 从高到低
        
        # 初始化
        current_image = image.clone()
        score_curve = []
        
        # 获取原始得分
        with torch.no_grad():
            output = self.model(current_image)
            initial_score = torch.softmax(output, dim=1)[0, target_class].item()
        score_curve.append(initial_score)
        
        # 逐步删除像素
        pixels_per_step = len(sorted_indices) // num_steps
        
        for step in range(1, num_steps):
            # 确定要删除的像素
            pixels_to_remove = sorted_indices[:step * pixels_per_step]
            
            # 创建mask
            mask = np.ones((img_h, img_w))
            mask_flat = mask.flatten()
            mask_flat[pixels_to_remove] = 0
            mask = mask_flat.reshape(img_h, img_w)
            
            # 应用高斯模糊到要删除的区域
            blurred_img = self._apply_blur(image, blur_sigma)
            
            # 混合原图和模糊图
            mask_tensor = torch.from_numpy(mask).float().to(self.device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            current_image = image * mask_tensor + blurred_img * (1 - mask_tensor)
            
            # 获取得分
            with torch.no_grad():
                output = self.model(current_image)
                score = torch.softmax(output, dim=1)[0, target_class].item()
            score_curve.append(score)
        
        # 计算AUC（使用梯形法则）
        deletion_auc = np.trapz(score_curve) / len(score_curve)
        
        return deletion_auc, score_curve
    
    def insertion_metric(self, image, heatmap, target_class, num_steps=50, blur_sigma=10):
        """
        Insertion指标
        
        原理：
        从模糊图像开始，逐步插入heatmap中最重要的像素，观察目标类得分上升速度
        好的解释应该导致得分快速上升
        
        参数:
        - image: [1, C, H, W] 原始图像tensor
        - heatmap: [H, W] 热力图，范围[0, 1]
        - target_class: 目标类别索引
        - num_steps: 插入步骤数
        - blur_sigma: 高斯模糊的sigma值
        
        返回:
        - insertion_auc: AUC值，越大越好
        - score_curve: 每一步的得分列表
        """
        self.model.eval()
        
        # 调整heatmap大小到图像尺寸
        img_h, img_w = image.shape[2], image.shape[3]
        heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
        
        # 按重要性排序像素位置
        flat_heatmap = heatmap_resized.flatten()
        sorted_indices = np.argsort(flat_heatmap)[::-1]  # 从高到低
        
        # 初始化为模糊图像
        blurred_img = self._apply_blur(image, blur_sigma)
        current_image = blurred_img.clone()
        score_curve = []
        
        # 获取初始得分（完全模糊）
        with torch.no_grad():
            output = self.model(current_image)
            initial_score = torch.softmax(output, dim=1)[0, target_class].item()
        score_curve.append(initial_score)
        
        # 逐步插入像素
        pixels_per_step = len(sorted_indices) // num_steps
        
        for step in range(1, num_steps):
            # 确定要插入的像素
            pixels_to_insert = sorted_indices[:step * pixels_per_step]
            
            # 创建mask
            mask = np.zeros((img_h, img_w))
            mask_flat = mask.flatten()
            mask_flat[pixels_to_insert] = 1
            mask = mask_flat.reshape(img_h, img_w)
            
            # 混合模糊图和原图
            mask_tensor = torch.from_numpy(mask).float().to(self.device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            current_image = image * mask_tensor + blurred_img * (1 - mask_tensor)
            
            # 获取得分
            with torch.no_grad():
                output = self.model(current_image)
                score = torch.softmax(output, dim=1)[0, target_class].item()
            score_curve.append(score)
        
        # 计算AUC
        insertion_auc = np.trapz(score_curve) / len(score_curve)
        
        return insertion_auc, score_curve
    
    def average_drop(self, image, heatmap, target_class, threshold=0.5):
        """
        Average Drop指标
        
        原理：
        只保留heatmap中大于阈值的重要区域，删除其余区域，计算得分下降
        
        参数:
        - image: [1, C, H, W] 原始图像tensor
        - heatmap: [H, W] 热力图，范围[0, 1]
        - target_class: 目标类别索引
        - threshold: 重要性阈值
        
        返回:
        - drop_percentage: 得分下降百分比
        - original_score: 原始得分
        - masked_score: mask后的得分
        """
        self.model.eval()
        
        # 获取原始得分
        with torch.no_grad():
            output = self.model(image)
            original_score = torch.softmax(output, dim=1)[0, target_class].item()
        
        # 调整heatmap大小
        img_h, img_w = image.shape[2], image.shape[3]
        heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
        
        # 创建mask（保留重要区域）
        mask = (heatmap_resized > threshold).astype(np.float32)
        
        # 应用mask（删除不重要区域）
        mask_tensor = torch.from_numpy(mask).float().to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        masked_image = image * mask_tensor
        
        # 获取mask后的得分
        with torch.no_grad():
            output = self.model(masked_image)
            masked_score = torch.softmax(output, dim=1)[0, target_class].item()
        
        # 计算下降百分比
        if original_score > 0:
            drop_percentage = (original_score - masked_score) / original_score * 100
        else:
            drop_percentage = 0
        
        return drop_percentage, original_score, masked_score
    
    def _apply_blur(self, image, sigma):
        """
        应用高斯模糊
        
        参数:
        - image: [1, C, H, W] tensor
        - sigma: 模糊程度
        
        返回: 模糊后的图像tensor
        """
        # 转换为numpy
        img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # 反归一化到[0, 1]
        img_np = img_np * 0.5 + 0.5
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(img_np, (0, 0), sigma)
        
        # 归一化回[-1, 1]
        blurred = (blurred - 0.5) / 0.5
        
        # 转换回tensor
        blurred_tensor = torch.from_numpy(blurred.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        return blurred_tensor


def evaluate_on_dataset(model, cam_generator, gradcam_generator, test_loader, 
                       device, num_samples=100, save_dir='./results/task2'):
    """
    在数据集上批量评估CAM和Grad-CAM
    
    参数:
    - model: 模型
    - cam_generator: CAM生成器
    - gradcam_generator: Grad-CAM生成器
    - test_loader: 测试数据加载器
    - device: 计算设备
    - num_samples: 评估样本数
    - save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = EvaluationMetrics(model, device)
    
    cam_results = {
        'deletion_auc': [],
        'insertion_auc': [],
        'average_drop': []
    }
    
    gradcam_results = {
        'deletion_auc': [],
        'insertion_auc': [],
        'average_drop': []
    }
    
    print(f"开始评估，样本数: {num_samples}")
    print("="*60)
    
    sample_count = 0
    for images, labels in tqdm(test_loader, desc="评估进度"):
        if sample_count >= num_samples:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        # 对batch中的每个图像
        for i in range(len(images)):
            if sample_count >= num_samples:
                break
            
            img_tensor = images[i:i+1]
            true_label = labels[i].item()
            
            # 生成CAM和Grad-CAM
            cam_heatmap, pred_class, _ = cam_generator.generate_cam(img_tensor)
            gradcam_heatmap, _, _ = gradcam_generator.generate_cam(img_tensor)
            
            # 只评估预测正确的样本
            if pred_class != true_label:
                continue
            
            # 评估CAM
            try:
                cam_del_auc, _ = evaluator.deletion_metric(img_tensor, cam_heatmap, pred_class, num_steps=30)
                cam_ins_auc, _ = evaluator.insertion_metric(img_tensor, cam_heatmap, pred_class, num_steps=30)
                cam_avg_drop, _, _ = evaluator.average_drop(img_tensor, cam_heatmap, pred_class)
                
                cam_results['deletion_auc'].append(cam_del_auc)
                cam_results['insertion_auc'].append(cam_ins_auc)
                cam_results['average_drop'].append(cam_avg_drop)
            except:
                continue
            
            # 评估Grad-CAM
            try:
                gradcam_del_auc, _ = evaluator.deletion_metric(img_tensor, gradcam_heatmap, pred_class, num_steps=30)
                gradcam_ins_auc, _ = evaluator.insertion_metric(img_tensor, gradcam_heatmap, pred_class, num_steps=30)
                gradcam_avg_drop, _, _ = evaluator.average_drop(img_tensor, gradcam_heatmap, pred_class)
                
                gradcam_results['deletion_auc'].append(gradcam_del_auc)
                gradcam_results['insertion_auc'].append(gradcam_ins_auc)
                gradcam_results['average_drop'].append(gradcam_avg_drop)
            except:
                continue
            
            sample_count += 1
    
    print(f"\n成功评估 {sample_count} 个样本")
    
    # 计算统计结果
    print("\n" + "="*60)
    print("评估结果统计")
    print("="*60)
    
    print("\nCAM:")
    print(f"  Deletion AUC (越小越好):    {np.mean(cam_results['deletion_auc']):.4f} ± {np.std(cam_results['deletion_auc']):.4f}")
    print(f"  Insertion AUC (越大越好):   {np.mean(cam_results['insertion_auc']):.4f} ± {np.std(cam_results['insertion_auc']):.4f}")
    print(f"  Average Drop (越大越好):    {np.mean(cam_results['average_drop']):.2f}% ± {np.std(cam_results['average_drop']):.2f}%")
    
    print("\nGrad-CAM:")
    print(f"  Deletion AUC (越小越好):    {np.mean(gradcam_results['deletion_auc']):.4f} ± {np.std(gradcam_results['deletion_auc']):.4f}")
    print(f"  Insertion AUC (越大越好):   {np.mean(gradcam_results['insertion_auc']):.4f} ± {np.std(gradcam_results['insertion_auc']):.4f}")
    print(f"  Average Drop (越大越好):    {np.mean(gradcam_results['average_drop']):.2f}% ± {np.std(gradcam_results['average_drop']):.2f}%")
    
    # 绘制对比图
    plot_comparison(cam_results, gradcam_results, save_dir)
    
    # 保存数值结果
    import json
    results_summary = {
        'CAM': {
            'deletion_auc_mean': float(np.mean(cam_results['deletion_auc'])),
            'deletion_auc_std': float(np.std(cam_results['deletion_auc'])),
            'insertion_auc_mean': float(np.mean(cam_results['insertion_auc'])),
            'insertion_auc_std': float(np.std(cam_results['insertion_auc'])),
            'average_drop_mean': float(np.mean(cam_results['average_drop'])),
            'average_drop_std': float(np.std(cam_results['average_drop']))
        },
        'Grad-CAM': {
            'deletion_auc_mean': float(np.mean(gradcam_results['deletion_auc'])),
            'deletion_auc_std': float(np.std(gradcam_results['deletion_auc'])),
            'insertion_auc_mean': float(np.mean(gradcam_results['insertion_auc'])),
            'insertion_auc_std': float(np.std(gradcam_results['insertion_auc'])),
            'average_drop_mean': float(np.mean(gradcam_results['average_drop'])),
            'average_drop_std': float(np.std(gradcam_results['average_drop']))
        }
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n结果已保存到: {save_dir}")
    print("="*60)


def plot_comparison(cam_results, gradcam_results, save_dir):
    """
    绘制CAM和Grad-CAM的对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['deletion_auc', 'insertion_auc', 'average_drop']
    titles = ['Deletion AUC (Lower is Better)', 'Insertion AUC (Higher is Better)', 'Average Drop % (Higher is Better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        cam_data = cam_results[metric]
        gradcam_data = gradcam_results[metric]
        
        # 箱线图
        axes[idx].boxplot([cam_data, gradcam_data], labels=['CAM', 'Grad-CAM'])
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)
        
        # 添加均值标记
        means = [np.mean(cam_data), np.mean(gradcam_data)]
        axes[idx].plot([1, 2], means, 'r*', markersize=15, label='Mean')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存: {os.path.join(save_dir, 'metrics_comparison.png')}")


def visualize_example(model, cam_generator, gradcam_generator, image, label, 
                     evaluator, save_path):
    """
    可视化一个样本的deletion和insertion曲线
    """
    device = next(model.parameters()).device
    
    # 生成热力图
    cam_heatmap, pred_class, _ = cam_generator.generate_cam(image)
    gradcam_heatmap, _, _ = gradcam_generator.generate_cam(image)
    
    # 计算指标
    cam_del_auc, cam_del_curve = evaluator.deletion_metric(image, cam_heatmap, pred_class, num_steps=50)
    cam_ins_auc, cam_ins_curve = evaluator.insertion_metric(image, cam_heatmap, pred_class, num_steps=50)
    
    gradcam_del_auc, gradcam_del_curve = evaluator.deletion_metric(image, gradcam_heatmap, pred_class, num_steps=50)
    gradcam_ins_auc, gradcam_ins_curve = evaluator.insertion_metric(image, gradcam_heatmap, pred_class, num_steps=50)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Deletion curve
    axes[0].plot(cam_del_curve, label=f'CAM (AUC={cam_del_auc:.3f})', linewidth=2)
    axes[0].plot(gradcam_del_curve, label=f'Grad-CAM (AUC={gradcam_del_auc:.3f})', linewidth=2)
    axes[0].set_xlabel('Deletion Steps', fontsize=12)
    axes[0].set_ylabel('Class Score', fontsize=12)
    axes[0].set_title('Deletion Metric', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Insertion curve
    axes[1].plot(cam_ins_curve, label=f'CAM (AUC={cam_ins_auc:.3f})', linewidth=2)
    axes[1].plot(gradcam_ins_curve, label=f'Grad-CAM (AUC={gradcam_ins_auc:.3f})', linewidth=2)
    axes[1].set_xlabel('Insertion Steps', fontsize=12)
    axes[1].set_ylabel('Class Score', fontsize=12)
    axes[1].set_title('Insertion Metric', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(root='./', train=False, download=False, 
                                    transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # 加载模型
    from models import VGG_CAM
    from cam_gradcam import CAM, GradCAM
    
    print('加载VGG_CAM模型...')
    model = VGG_CAM().to(device)
    
    checkpoint_path = './checkpoint/VGG_CAM/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功，准确率: {checkpoint['accuracy']:.2f}%")
    else:
        print("警告: 未找到预训练模型，请先训练模型")
        return
    
    # 初始化CAM和Grad-CAM
    cam_generator = CAM(model)
    target_layer = model.layer5[-1]
    gradcam_generator = GradCAM(model, target_layer=target_layer)
    
    # 批量评估
    evaluate_on_dataset(model, cam_generator, gradcam_generator, test_loader, 
                       device, num_samples=100, save_dir='./results/task2')
    
    print("\nTask 2 完成！")


if __name__ == '__main__':
    main()
