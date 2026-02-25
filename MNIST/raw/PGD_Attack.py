import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --------------------------
# 配置区域
# --------------------------
EPSILON = 2/255  
ALPHA = 0.5/255
ITERS = 6
BATCH_SIZE = 1 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'raw/model/mnist_adv_trained_best.pkl'
DATA_ROOT = './raw/data'       

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(320, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x) 
        
        return x

# 定义归一化参数 (转为Tensor以便在GPU上计算)
mean_list = [0.485, 0.456, 0.406]
std_list = [0.229, 0.224, 0.225]

# --------------------------
# 核心：PGD 攻击函数 (修正版)
# --------------------------
def pgd_attack(model, images_raw, labels, epsilon, alpha, iters):
    """
    PGD 攻击函数
    注意：输入的 images_raw 必须是 [0, 1] 范围的原始图片，未归一化！
    """
    # 原始图片 [0, 1]
    original_images = images_raw.clone().detach()
    
    # 1. 随机初始化 (Random Start)
    # 在原始图片基础上加一点 [-eps, eps] 的随机噪声
    adv_images = images_raw.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1) # 确保还在 [0, 1] 范围内
    
    # 2. 迭代攻击
    for _ in range(iters):
        
        # 锁死模型参数，确保只对输入求梯度
        for param in model.parameters():
            param.requires_grad = False 
            
        adv_images.requires_grad = True
        
        # --- 攻击时，先归一化，再喂给模型 ---
        outputs = model((adv_images))
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 获取梯度
        grad = adv_images.grad.detach()
        
        # 更新对抗样本 (PGD Step)
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # 投影 (Projection): 确保扰动不超过 Epsilon
        eta = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = original_images + eta
        
        # 裁剪 (Clipping): 确保像素值在 [0, 1] 合法范围内
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
    return adv_images

def main():
    print(f"正在复现 PGD 攻击，Epsilon = {EPSILON:.4f}，Alpha = {ALPHA:.4f}，Iters = {ITERS}...")
    
    # 1. 加载数据
    if not os.path.exists(os.path.join(DATA_ROOT, 'MNIST')):
        print(f"错误：没有找到 {DATA_ROOT}/MNIST 目录")
        return

    test_loader = DataLoader(
        datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=False)
    
    
    # 2. 加载模型
    print("正在初始化 CNN 模型...")
    model = MnistModel().to(DEVICE)
    print(f"正在加载参数: {CHECKPOINT_PATH}")
    try:
        # 加载参数字典
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        # 将参数加载进模型
        model.load_state_dict(state_dict)
        print("参数加载成功！")
    except Exception as e:
        print(f"参数加载失败: {e}")
        return
    
    model.eval()  # 固定 BatchNorm 和 Dropout
    
    correct = 0
    total = 0
    adv_success = 0
    
    # 只跑前 2000 张，节省时间
    max_samples = 2000
    pbar = tqdm(total=max_samples)  # 进度条显示
    

    for i, (data, target) in enumerate(test_loader):
        if total >= max_samples: break
        
        data, target = data.to(DEVICE), target.to(DEVICE) # data 此时是归一化过的
        
        # 3. 预测原始准确率
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        
        if init_pred.item() != target.item():
            continue # 如果原本就错了，跳过
             
        
        # 4. 执行 PGD 攻击 (传入 [0, 1] 的图片)
        # 注意：这里不需要 squeeze，保持 [B, C, H, W] 4维维度更安全
        perturbed_data_raw = pgd_attack(model, data, target, EPSILON, ALPHA, ITERS)
        
        # 5. 再次预测 (预测前要重新归一化)
        final_output = model(perturbed_data_raw)
        final_pred = final_output.max(1, keepdim=True)[1]
        
        total += 1
        pbar.update(1)
        
        if final_pred.item() != target.item():
            adv_success += 1
            
            # 画图 (只画前 3 张)
            if 2 <= adv_success <= 4:
                print(f"\n攻击成功！真值: {target.item()} -> 攻击后: {final_pred.item()}")
                
                # 准备显示用的 numpy 数据
                img_orig = data[0].cpu().numpy().transpose(1, 2, 0)
                img_adv = perturbed_data_raw[0].cpu().numpy().transpose(1, 2, 0)
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title(f"Original: {target.item()}")
                plt.imshow(img_orig)
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.title(f"PGD Attack: {final_pred.item()}")
                plt.imshow(img_adv)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'pgd_example_{adv_success}.png') # 保存而不是 show，防止卡住
                plt.show() 
                plt.close() # 关闭图像，释放内存

    pbar.close()
    print(f"\n攻击完成！样本数: {total}")
    print(f"攻击成功数 (Adv Success): {adv_success}")
    print(f"攻击成功率 (Error Rate): {adv_success/total:.2%}")
    print(f"模型鲁棒准确率 (Robust Acc): {1 - adv_success/total:.2%}")

if __name__ == '__main__':
    main()