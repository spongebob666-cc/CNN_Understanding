import torch
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    device = torch.device('cpu')
    model.to(device)
    # create dataset, data augmentation
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
    checkpoint_dir = f'./checkpoint/{args.model}'
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    train_dataset = datasets.CIFAR10(root='./', train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./', train=False, download=False, transform=test_transform)
    # create dataloader
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # create scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ctreat summary writer
    log_dir = f'{args.logdir}/{args.model}'
    writer = SummaryWriter(log_dir)

    criterion = torch.nn.CrossEntropyLoss()
        # train
    best_acc = 0.0
    for epoch in range(args.epochs):
            # get the inputs; data is a list of [inputs, labels]
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += labels.size(0)
            epoch_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {epoch_loss / (batch_idx+1):.4f}, Acc: {100.*epoch_correct/epoch_total:.2f}%')
        # scheduler adjusts learning rate
        scheduler.step()
        # log
        train_loss = epoch_loss / len(trainloader)
        train_acc = 100.*epoch_correct/epoch_total

        # test
        test_loss, test_acc = test_model(model, testloader, device, criterion)
        writer.add_scalar('Loss/Train', train_loss, epoch+1)
        writer.add_scalar('Loss/Test', test_loss, epoch+1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/Test', test_acc, epoch+1)

        print(f'Epoch [{epoch+1}/{args.epochs}] Summary: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, best_acc,checkpoint_path)
    writer.close()

def test_model(model, testloader, device, criterion):
    model.eval()
    model.to(device)
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

def save_checkpoint(model, optimizer, epoch, acc,path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': acc
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    acc = checkpoint['accuracy']
    return model, optimizer, epoch, acc

def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    device = torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    model_, optimizer_, epoch_, acc_ = load_checkpoint (model, optimizer, args.checkpoint_path)
    print(f'Loaded checkpoint from epoch {epoch_} with accuracy {acc_:.2f}%')

    # create testing dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='./', train=False, download=False, transform=test_transform)
    # create dataloader
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # test
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = test_model(model, dataloader, device, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--logdir', type=str, default='./logs', help='directory to save logs')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/best_model.pth', help='path to save/load checkpoint')
    parser.add_argument('--model', type=str, default='ResNet', choices=['VGG', 'ResNet', 'ResNext'], help='model to use')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test the model')
    args = parser.parse_args()

    if args.model == 'VGG':
        model = VGG()
    elif args.model == 'ResNet':
        model = ResNet()
    elif args.model == 'ResNext':
        model = ResNext()
    
    if args.mode == 'train':
        train(model, args)
    elif args.mode == 'test':
        test(model, args)
    # train / test
