import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.readData import read_dataset
from utils.ResNet import ResNet18

# 1. 定义类别名称 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def inverse_normalize(tensor, mean, std):
    """
    反归一化：将 Tensor 从 (input - mean) / std 还原回原始图像
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_results(model, loader, device, num_images=16):
    """
    可视化函数：抽取 batch 并画图
    """
    model.eval() # 确保模型处于评估模式
    
    # 获取一个 Batch 的数据
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # 搬运到设备上进行预测
    images_device = images.to(device)
    outputs = model(images_device)
    _, preds = torch.max(outputs, 1)
    
    # 准备画图
    fig = plt.figure(figsize=(12, 12))
    
    # 定义反归一化的参数 (和 readData.py 里保持一致)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print(f"\n--- 正在抽取 {num_images} 张测试集图片进行“抽查” ---")

    for idx in range(num_images):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        
        # 处理单张图片用于显示
        # 1. 克隆一份并在 CPU 上操作
        img = images[idx].clone().cpu() 
        # 2. 反归一化
        img = inverse_normalize(img, mean, std)
        # 3. 转换维度 (C, H, W) -> (H, W, C)
        img = img.numpy().transpose((1, 2, 0))
        # 4. 限制范围在 0-1 之间 (防止噪点)
        img = np.clip(img, 0, 1)
        
        # 显示图片
        ax.imshow(img)
        
        # 获取标签名称
        true_label = classes[labels[idx]]
        pred_label = classes[preds[idx]]
        
        # 设置标题颜色：预测正确=绿色，错误=红色
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.tight_layout()
    plt.show()

def main():
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_class = 10
    batch_size = 100
    
    print(f"正在使用设备: {device}")
    
    # 加载数据
    _, _, test_loader = read_dataset(batch_size=batch_size, pic_path='cifar10/dataset')
    
    # 重建模型结构
    model = ResNet18()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, n_class)
    model = model.to(device)

    # 载入权重
    checkpoint_path = 'cifar10/checkpoint/resnet18_cifar10_trades_best.pt'
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ 成功加载模型权重！")
    except FileNotFoundError:
        print(f"❌ 找不到权重文件: {checkpoint_path}")
        return

    # ---------------------------
    # 第一部分：计算整体准确率
    # ---------------------------
    total_sample = 0
    right_sample = 0
    model.eval() 
    
    print("正在计算测试集整体准确率...")
    
    with torch.no_grad(): # 加上 no_grad 省显存
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            _, pred = torch.max(output, 1)    
            
            correct_tensor = pred.eq(target.data.view_as(pred))
            
            # 用 target.size(0) 更稳健
            total_sample += target.size(0) 
            
            for i in correct_tensor:
                if i:
                    right_sample += 1
                    
    acc = 100 * right_sample / total_sample
    print(f"🏆 测试集最终准确率 (Accuracy): {acc:.2f}%")

    # ---------------------------
    # 第二部分：可视化抽查
    # ---------------------------
    # 随机抽查 16 张图看看效果
    visualize_results(model, test_loader, device, num_images=16)

if __name__ == '__main__':
    main()