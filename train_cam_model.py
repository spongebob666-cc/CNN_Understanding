"""
train_cam_model.py - 训练支持CAM的模型

用于训练VGG_CAM和ResNet_CAM模型
"""

import torch
import torch.nn.functional as F
from models import VGG_CAM, ResNet_CAM
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse


def train_cam_model(model, args):
    """
    训练CAM版本的模型
    
    参数:
    - model: VGG_CAM或ResNet_CAM模型
    - args: 训练配置参数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    model.to(device)
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 数据加载
    print('加载CIFAR-10数据集...')
    train_dataset = datasets.CIFAR10(root='./', train=True, download=False, 
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./', train=False, download=False, 
                                    transform=test_transform)
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                           shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=2)
    
    # 优化器和调度器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                               momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Tensorboard
    log_dir = f"./logs/{args.model_name}"
    writer = SummaryWriter(log_dir)
    print(f'Tensorboard日志保存在: {log_dir}')
    
    # 检查点目录
    checkpoint_dir = f"./checkpoint/{args.model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    
    print(f'\n开始训练 {args.model_name}...')
    print(f'总轮次: {args.epochs}, 批大小: {args.batch_size}, 学习率: {args.lr}')
    print('='*60)
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += labels.size(0)
            epoch_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], '
                      f'Step [{batch_idx+1}/{len(trainloader)}], '
                      f'Loss: {epoch_loss / (batch_idx+1):.4f}, '
                      f'Acc: {100.*epoch_correct/epoch_total:.2f}%')
        
        scheduler.step()
        
        # 验证
        train_loss = epoch_loss / len(trainloader)
        train_acc = 100.*epoch_correct/epoch_total
        test_loss, test_acc = test_model(model, testloader, device, criterion)
        
        # 记录
        writer.add_scalar('Loss/Train', train_loss, epoch+1)
        writer.add_scalar('Loss/Test', test_loss, epoch+1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/Test', test_acc, epoch+1)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch+1)
        
        print(f'Epoch [{epoch+1}/{args.epochs}] 总结: '
              f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | '
              f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
        print('-'*60)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': best_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'✓ 保存最佳模型，准确率: {best_acc:.2f}%')
            print('-'*60)
    
    writer.close()
    print('\n' + '='*60)
    print(f'训练完成！最佳测试准确率: {best_acc:.2f}%')
    print(f'模型已保存到: {checkpoint_path}')
    print('='*60)


def test_model(model, testloader, device, criterion):
    """
    测试模型
    
    返回:
    - avg_test_loss: 平均测试损失
    - test_acc: 测试准确率（百分比）
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    avg_test_loss = test_loss / len(testloader)
    test_acc = 100.*test_correct/test_total
    return avg_test_loss, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练支持CAM的模型')
    parser.add_argument('--model', type=str, default='VGG_CAM', 
                       choices=['VGG_CAM', 'ResNet_CAM'],
                       help='选择模型: VGG_CAM 或 ResNet_CAM')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='学习率 (默认: 0.1)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮次 (默认: 50)')
    
    args = parser.parse_args()
    args.model_name = args.model
    
    # 创建模型
    if args.model == 'VGG_CAM':
        model = VGG_CAM()
        print('创建VGG_CAM模型')
    elif args.model == 'ResNet_CAM':
        model = ResNet_CAM()
        print('创建ResNet_CAM模型')
    
    # 训练
    train_cam_model(model, args)
