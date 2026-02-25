import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.readData import read_dataset  

# CIFAR-10 的 10 个类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    """
    用于可视化图片的辅助函数
    因为数据经过了 Normalize((0.485,...), (0.229,...))
    所以显示前需要把这个过程逆转回来，否则图片会黑乎乎或颜色怪异。
    """
    # 1. 转回 numpy
    img = img.numpy().transpose((1, 2, 0)) # 把 (C,H,W) 变成 (H,W,C)
    
    # 2. 反归一化 (Un-normalize)
    # input = (output - mean) / std  ==>  output = input * std + mean
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    
    # 3. 修正范围到 [0, 1] 之间，防止噪点报错
    img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    plt.axis('off') # 不显示坐标轴
    plt.show()

def main():
    print("Step 1: 正在尝试加载/下载数据集...")
    # 我们设置 batch_size=16，即一次拿16张图来看
    try:
        train_loader, valid_loader, test_loader = read_dataset(batch_size=16, pic_path='./dataset')
        print("✅ 数据集加载成功！")
        print(f"训练集 Batch 数量: {len(train_loader)}")
        print(f"验证集 Batch 数量: {len(valid_loader)}")
        print(f"测试集 Batch 数量: {len(test_loader)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    print("\nStep 2: 正在抽取一个 Batch 的图片进行可视化...")
    # 从 train_loader 中获取第一批数据
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(f"这批图片的 Tensor 形状: {images.shape}") 
    # 应该是 [16, 3, 32, 32] -> 16张图, 3通道, 32x32大小

    # 打印标签
    print("对应的标签: ", " ".join(f'{classes[labels[j]]:5s}' for j in range(16)))

    # 拼接成一张大图并显示
    print("\n请看弹出的图片窗口（注意观察图片里有没有被挖掉的黑块 - Cutout效果）")
    imshow(make_grid(images, nrow=4, padding=2)) # nrow=4 表示一行显示4张

if __name__ == '__main__':
    main()