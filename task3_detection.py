"""
task3_detection.py - Task 3: Grad-CAM for Object Detection

将Grad-CAM应用到目标检测任务（Faster R-CNN）
展示在非分类任务上的可视化解释
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os


class GradCAM_Detection:
    """
    为目标检测模型定制的Grad-CAM
    
    关键修改：
    1. 目标不是类别logit，而是特定检测框的得分
    2. 需要选择backbone中的某一层作为目标层
    3. 可以为每个检测框生成单独的热力图
    """
    
    def __init__(self, model, target_layer):
        """
        参数:
        - model: Faster R-CNN模型
        - target_layer: backbone中的目标卷积层
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
        
    def _register_hooks(self):
        """注册hooks捕获特征图和梯度"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam_for_detection(self, image_tensor, box_idx=0):
        """
        为特定检测框生成Grad-CAM
        
        参数:
        - image_tensor: [C, H, W] 输入图像tensor
        - box_idx: 检测框索引（按置信度排序）
        
        返回:
        - cam: 热力图 [H, W]
        - detection: 检测结果字典 {'box': [x1,y1,x2,y2], 'label': int, 'score': float}
        """
        self.model.eval()
        
        # 准备输入
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        image_tensor.requires_grad = True
        
        # 前向传播
        with torch.set_grad_enabled(True):
            predictions = self.model(image_tensor)
        
        pred = predictions[0]
        
        # 检查是否有检测结果
        if len(pred['boxes']) == 0:
            print("未检测到任何对象")
            return None, None
        
        # 选择目标检测框
        if box_idx >= len(pred['boxes']):
            box_idx = 0
        
        target_box = pred['boxes'][box_idx]
        target_label = pred['labels'][box_idx]
        target_score = pred['scores'][box_idx]
        
        # 反向传播（对检测得分）
        self.model.zero_grad()
        target_score.backward()
        
        # 计算Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            print("未能捕获梯度或激活值")
            return None, None
        
        # 计算权重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = torch.relu(cam)
        
        # 归一化
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        # 构造检测结果
        detection = {
            'box': target_box.detach().cpu().numpy(),
            'label': target_label.item(),
            'score': target_score.item()
        }
        
        return cam, detection


def load_image(image_path):
    """
    加载图像并转换为tensor
    
    返回:
    - image_tensor: [C, H, W] tensor
    - original_image: PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image)
    
    return image_tensor, image


def visualize_detection_with_cam(image, detection, cam, class_names, save_path=None):
    """
    可视化检测结果和Grad-CAM
    
    参数:
    - image: PIL Image 或 numpy array
    - detection: 检测结果字典
    - cam: 热力图 [H, W]
    - class_names: 类别名称列表
    - save_path: 保存路径
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    
    # 调整CAM大小
    cam_resized = cv2.resize(cam, (w, h))
    
    # 应用颜色映射
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlayed = cam_colored * 0.4 + image * 0.6
    overlayed = np.uint8(overlayed)
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像 + 检测框
    axes[0].imshow(image)
    box = detection['box']
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    label_text = f"{class_names[detection['label']]}: {detection['score']:.2f}"
    axes[0].text(box[0], box[1]-5, label_text, color='red', fontsize=12, 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_title('Detection Result', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM叠加
    axes[1].imshow(overlayed)
    rect2 = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                              linewidth=2, edgecolor='yellow', facecolor='none')
    axes[1].add_patch(rect2)
    axes[1].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 纯热力图
    axes[2].imshow(cam_resized, cmap='jet')
    axes[2].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存到: {save_path}")
    
    plt.close()


def demo_on_images(image_dir, save_dir='./results/task3'):
    """
    在一组图像上演示Grad-CAM for Detection
    
    参数:
    - image_dir: 图像目录
    - save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载预训练的Faster R-CNN模型
    print("加载Faster R-CNN模型...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 选择target layer（backbone的最后一层）
    target_layer = model.backbone.body.layer4[-1].conv3
    
    # 初始化Grad-CAM
    gradcam = GradCAM_Detection(model, target_layer)
    
    # COCO类别名称（简化版）
    class_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 处理图像
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 张图像")
    
    for img_file in image_files:
        print(f"\n处理: {img_file}")
        img_path = os.path.join(image_dir, img_file)
        
        # 加载图像
        image_tensor, original_image = load_image(img_path)
        image_tensor = image_tensor.to(device)
        
        # 生成多个检测框的Grad-CAM
        # 先做一次检测看有多少个框
        with torch.no_grad():
            predictions = model([image_tensor])
        
        num_detections = len(predictions[0]['boxes'])
        print(f"  检测到 {num_detections} 个对象")
        
        # 为top-3检测框生成Grad-CAM
        for box_idx in range(min(3, num_detections)):
            cam, detection = gradcam.generate_cam_for_detection(image_tensor, box_idx=box_idx)
            
            if cam is not None and detection is not None:
                save_path = os.path.join(save_dir, f"{os.path.splitext(img_file)[0]}_box{box_idx}.png")
                visualize_detection_with_cam(original_image, detection, cam, class_names, save_path)
    
    print(f"\n所有结果已保存到: {save_dir}")


def main():
    """主函数"""
    print("="*60)
    print("Task 3: Grad-CAM for Object Detection")
    print("="*60)
    
    # 示例：在测试图像上运行
    # 你需要准备一些测试图像
    image_dir = './test_images'
    
    if not os.path.exists(image_dir):
        print(f"\n请在 {image_dir} 目录下放置测试图像")
        print("可以从互联网下载一些包含常见物体（人、车、动物等）的图像")
        print("\n创建示例目录...")
        os.makedirs(image_dir, exist_ok=True)
        print(f"目录已创建: {image_dir}")
        print("请添加图像后重新运行此脚本")
        return
    
    demo_on_images(image_dir, save_dir='./results/task3')
    
    print("\nTask 3 完成！")


if __name__ == '__main__':
    main()


"""
Task 3 实现说明和扩展方向：

1. 当前实现：
   - 使用预训练的Faster R-CNN模型
   - 为每个检测框生成Grad-CAM热力图
   - 可视化检测结果和模型关注区域

2. 观察要点：
   - 热力图是否集中在检测框内？
   - 对于多个重叠的检测框，热力图有何不同？
   - 错误检测的热力图有什么特征？

3. 可能的failure cases：
   - 小物体：热力图可能扩散到周围区域
   - 遮挡物体：模型可能关注可见部分或周围上下文
   - 相似物体：模型可能关注区分性特征
   - 背景复杂：热力图可能受背景干扰

4. 扩展方向：
   a) 语义分割（DeepLabV3）：
      - 为每个类别生成Grad-CAM
      - 对比不同类别的关注区域
   
   b) 实例分割（Mask R-CNN）：
      - 结合mask和Grad-CAM
      - 分析模型如何区分不同实例
   
   c) 定量评估：
      - IoU between heatmap and ground truth box
      - Pointing game: 热力图峰值是否在物体内
      - Energy-based pointing game
   
   d) 多层分析：
      - 对比不同层的Grad-CAM
      - 分析从低级特征到高级语义的过渡

5. 改进方法：
   - Grad-CAM++: 更好地处理多个实例
   - Score-CAM: 不依赖梯度，更稳定
   - LayerCAM: 更精细的空间定位
"""
