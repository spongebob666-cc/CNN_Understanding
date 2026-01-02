"""
run_all_tasks.py - 运行所有任务的主脚本

按顺序执行：
1. Task 1: 训练模型并生成CAM/Grad-CAM可视化
2. Task 2: 评估指标
3. Task 3: 目标检测应用（可选）
"""

import os
import sys
import argparse


def print_banner(text):
    """打印美化的横幅"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_task1(args):
    """运行Task 1"""
    print_banner("Task 1: CAM和Grad-CAM实现")
    
    # 检查是否已有训练好的模型
    checkpoint_path = f'./checkpoint/{args.model}/best_model.pth'
    
    if not os.path.exists(checkpoint_path) or args.retrain:
        print("步骤 1/2: 训练CAM模型...")
        print(f"模型: {args.model}")
        print(f"训练轮次: {args.epochs}")
        print(f"批大小: {args.batch_size}")
        print(f"学习率: {args.lr}\n")
        
        # 训练模型
        from train_cam_model import train_cam_model
        from models import VGG_CAM, ResNet_CAM
        
        if args.model == 'VGG_CAM':
            model = VGG_CAM()
        else:
            model = ResNet_CAM()
        
        train_args = argparse.Namespace(
            model_name=args.model,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs
        )
        
        train_cam_model(model, train_args)
    else:
        print(f"找到已训练的模型: {checkpoint_path}")
        print("跳过训练步骤（使用 --retrain 强制重新训练）\n")
    
    print("步骤 2/2: 生成可视化...")
    
    # 生成可视化
    from cam_gradcam import main as cam_main
    cam_main()
    
    print("\n✓ Task 1 完成！")
    print(f"结果保存在: ./results/task1_{args.model.lower()}/")


def run_task2(args):
    """运行Task 2"""
    print_banner("Task 2: 评估指标实现")
    
    # 检查模型
    checkpoint_path = f'./checkpoint/{args.model}/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"错误: 未找到模型 {checkpoint_path}")
        print("请先运行Task 1训练模型")
        return
    
    print(f"使用模型: {args.model}")
    print(f"评估样本数: {args.num_eval_samples}\n")
    
    # 运行评估
    from task2_evaluation import main as task2_main
    task2_main()
    
    print("\n✓ Task 2 完成！")
    print("结果保存在: ./results/task2/")


def run_task3(args):
    """运行Task 3"""
    print_banner("Task 3: 目标检测应用")
    
    # 检查测试图像目录
    if not os.path.exists('./test_images'):
        print("创建测试图像目录: ./test_images/")
        os.makedirs('./test_images', exist_ok=True)
        print("\n请在 ./test_images/ 目录下放置测试图像后重新运行")
        print("建议使用包含常见物体（人、车、动物等）的图像")
        return
    
    image_files = [f for f in os.listdir('./test_images') 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(image_files) == 0:
        print("错误: ./test_images/ 目录中没有图像")
        print("请添加测试图像（.jpg, .png, .jpeg格式）")
        return
    
    print(f"找到 {len(image_files)} 张测试图像\n")
    
    # 运行检测
    from task3_detection import main as task3_main
    task3_main()
    
    print("\n✓ Task 3 完成！")
    print("结果保存在: ./results/task3/")


def main():
    parser = argparse.ArgumentParser(description='运行所有任务')
    
    # 通用参数
    parser.add_argument('--tasks', type=str, default='all',
                       help='要运行的任务: all, 1, 2, 3, 或组合如 "1,2"')
    parser.add_argument('--model', type=str, default='VGG_CAM',
                       choices=['VGG_CAM', 'ResNet_CAM'],
                       help='模型类型')
    
    # Task 1参数
    parser.add_argument('--retrain', action='store_true',
                       help='强制重新训练模型（即使已有checkpoint）')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='学习率')
    
    # Task 2参数
    parser.add_argument('--num_eval_samples', type=int, default=100,
                       help='Task 2评估样本数')
    
    args = parser.parse_args()
    
    # 解析要运行的任务
    if args.tasks == 'all':
        tasks_to_run = ['1', '2', '3']
    else:
        tasks_to_run = args.tasks.split(',')
    
    print_banner("深度卷积网络可视化解释 - CAM & Grad-CAM")
    print(f"要运行的任务: {', '.join(['Task ' + t for t in tasks_to_run])}")
    print(f"模型: {args.model}\n")
    
    # 按顺序运行任务
    try:
        if '1' in tasks_to_run:
            run_task1(args)
        
        if '2' in tasks_to_run:
            run_task2(args)
        
        if '3' in tasks_to_run:
            run_task3(args)
        
        print_banner("所有任务完成！")
        print("结果总结:")
        print("  Task 1: ./results/task1_*/")
        print("  Task 2: ./results/task2/")
        print("  Task 3: ./results/task3/")
        print("\n查看 README.md 了解更多详情")
        
    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


"""
使用示例:

1. 运行所有任务（使用默认参数）:
   python run_all_tasks.py

2. 只运行Task 1:
   python run_all_tasks.py --tasks 1

3. 运行Task 1和2:
   python run_all_tasks.py --tasks 1,2

4. 使用ResNet_CAM模型:
   python run_all_tasks.py --model ResNet_CAM

5. 快速训练（减少轮次）:
   python run_all_tasks.py --epochs 20

6. 强制重新训练:
   python run_all_tasks.py --retrain

7. 完整自定义:
   python run_all_tasks.py --model VGG_CAM --epochs 50 --batch_size 64 --lr 0.1 --tasks all
"""
