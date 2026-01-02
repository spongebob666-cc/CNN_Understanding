"""
train_cam_model_gpu.py - GPU优化版训练脚本

相比原版train_cam_model.py的改进：
1. 支持多GPU训练（DataParallel）
2. 增加混合精度训练（节省显存，加速训练）
3. 优化数据加载（pin_memory, prefetch）
4. 显示GPU使用情况
5. 更灵活的设备选择
"""

import torch
import torch.nn.functional as F
from models import VGG_CAM, ResNet_CAM
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse


def get_gpu_info():
    """显示GPU信息"""
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print(f"GPU信息:")
        print(f"{'='*60}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"{'='*60}\n")
    else:
        print("警告: 未检测到GPU，将使用CPU训练（速度较慢）")


def train_cam_model(model, args):
    """
    GPU优化的训练函数
    """
    # 显示GPU信息
    get_gpu_info()
    
    # 设备选择
    if args.gpu_ids and torch.cuda.is_available():
        # 指定GPU
        device = torch.device(f'cuda:{args.gpu_ids[0]}')
        print(f'使用指定GPU: {args.gpu_ids}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
    
    model = model.to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练!")
        if args.gpu_ids:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        else:
            model = torch.nn.DataParallel(model)
    
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
    
    # 数据加载（GPU优化）
    print('加载CIFAR-10数据集...')
    train_dataset = datasets.CIFAR10(root='./', train=True, download=False, 
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./', train=False, download=False, 
                                    transform=test_transform)
    
    # GPU优化的DataLoader设置
    num_workers = args.num_workers if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()  # GPU时启用pin_memory加速
    
    trainloader = DataLoader(train_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=True, 
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           persistent_workers=num_workers > 0)
    
    testloader = DataLoader(test_dataset, 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          persistent_workers=num_workers > 0)
    
    # 优化器和调度器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                               momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 混合精度训练（如果支持）
    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        print("启用混合精度训练（AMP）- 节省显存并加速")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Tensorboard
    log_dir = f"./logs/{args.model_name}"
    writer = SummaryWriter(log_dir)
    print(f'Tensorboard日志: {log_dir}')
    
    # 检查点目录
    checkpoint_dir = f"./checkpoint/{args.model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    
    print(f'\n开始训练 {args.model_name}...')
    print(f'总轮次: {args.epochs}, 批大小: {args.batch_size}, 学习率: {args.lr}')
    print(f'使用混合精度: {use_amp}')
    print('='*60)
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += labels.size(0)
            epoch_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                # 显示GPU显存使用（如果使用GPU）
                gpu_mem_info = ""
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    gpu_mem_max = torch.cuda.max_memory_allocated() / 1024**3
                    gpu_mem_info = f", GPU显存: {gpu_mem:.1f}/{gpu_mem_max:.1f}GB"
                
                print(f'Epoch [{epoch+1}/{args.epochs}], '
                      f'Step [{batch_idx+1}/{len(trainloader)}], '
                      f'Loss: {epoch_loss / (batch_idx+1):.4f}, '
                      f'Acc: {100.*epoch_correct/epoch_total:.2f}%'
                      f'{gpu_mem_info}')
        
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
        
        if torch.cuda.is_available():
            writer.add_scalar('GPU_Memory_GB', torch.cuda.memory_allocated() / 1024**3, epoch+1)
        
        print(f'Epoch [{epoch+1}/{args.epochs}] 总结: '
              f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | '
              f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
        print('-'*60)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            
            # 如果使用DataParallel，保存原始模型
            model_to_save = model.module if hasattr(model, 'module') else model
            
            checkpoint = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': best_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'✓ 保存最佳模型，准确率: {best_acc:.2f}%')
            print('-'*60)
    
    writer.close()
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print('\n' + '='*60)
    print(f'训练完成！最佳测试准确率: {best_acc:.2f}%')
    print(f'模型已保存到: {checkpoint_path}')
    print('='*60)


def test_model(model, testloader, device, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
    parser = argparse.ArgumentParser(description='GPU优化的CAM模型训练')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='VGG_CAM', 
                       choices=['VGG_CAM', 'ResNet_CAM'],
                       help='模型类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批大小（GPU可以用更大的值，如256或512）')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮次')
    
    # GPU参数
    parser.add_argument('--multi_gpu', action='store_true',
                       help='使用多GPU训练（如果有多个GPU）')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                       help='指定GPU ID，例如: --gpu_ids 0 1 2')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练（节省显存，加速训练）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载的worker数量（GPU推荐4-8）')
    
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
