import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.readData import read_dataset
from utils.ResNet import ResNet18

# --------------------------
# 配置区域
# --------------------------
EPSILON = 8/255  
ALPHA = 2/255
ITERS = 7
BATCH_SIZE = 1 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'cifar10/checkpoint/resnet18_cifar10_trades_best.pt'

# 定义归一化参数 (转为Tensor以便在GPU上计算)
mean_list = [0.485, 0.456, 0.406]
std_list = [0.229, 0.224, 0.225]

# --------------------------
# 辅助函数：手动归一化与反归一化
# --------------------------
def normalize_tensor(batch_img):
    """ 将 [0, 1] 的 Tensor 归一化 (减均值除方差) 用于喂给模型 """
    # batch_img shape: [B, 3, H, W]
    mean = torch.tensor(mean_list).view(1, 3, 1, 1).to(batch_img.device)
    std = torch.tensor(std_list).view(1, 3, 1, 1).to(batch_img.device)
    return (batch_img - mean) / std

def unnormalize_tensor(batch_img):
    """ 将归一化的 Tensor 还原回 [0, 1] 用于 PGD 更新和显示 """
    mean = torch.tensor(mean_list).view(1, 3, 1, 1).to(batch_img.device)
    std = torch.tensor(std_list).view(1, 3, 1, 1).to(batch_img.device)
    return batch_img * std + mean

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
        
        # --- 关键点：攻击时，先归一化，再喂给模型 ---
        outputs = model(normalize_tensor(adv_images))
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 获取梯度
        grad = adv_images.grad.detach()
        
        # 更新对抗样本 (PGD Step)
        adv_images = adv_images + alpha * grad.sign()
        
        # 投影 (Projection): 确保扰动不超过 Epsilon
        eta = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = original_images + eta
        
        # 裁剪 (Clipping): 确保像素值在 [0, 1] 合法范围内
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
    return adv_images

def main():
    print(f"正在复现 PGD 攻击，Epsilon = {EPSILON:.4f}，Alpha = {ALPHA:.4f}，Iters = {ITERS}...")
    
    # 1. 加载数据
    _, _, test_loader = read_dataset(batch_size=BATCH_SIZE, pic_path='cifar10/dataset')
    
    # 2. 加载模型
    model = ResNet18()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # 固定 BatchNorm 和 Dropout
    
    correct = 0
    total = 0
    adv_success = 0
    
    # 只跑前 2000 张，节省时间
    max_samples = 2000
    pbar = tqdm(total=max_samples)  # 进度条显示
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i, (data, target) in enumerate(test_loader):
        if total >= max_samples: break
        
        data, target = data.to(DEVICE), target.to(DEVICE) # data 此时是归一化过的
        
        # 3. 预测原始准确率
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        
        if init_pred.item() != target.item():
            continue # 如果原本就错了，跳过
            
        # --- 关键步骤：先反归一化回 [0, 1] ---
        data_raw = unnormalize_tensor(data)
        
        # 4. 执行 PGD 攻击 (传入 [0, 1] 的图片)
        # 注意：这里不需要 squeeze，保持 [B, C, H, W] 4维维度更安全
        perturbed_data_raw = pgd_attack(model, data_raw, target, EPSILON, ALPHA, ITERS)
        
        # 5. 再次预测 (预测前要重新归一化)
        final_output = model(normalize_tensor(perturbed_data_raw))
        final_pred = final_output.max(1, keepdim=True)[1]
        
        total += 1
        pbar.update(1)
        
        if final_pred.item() != target.item():
            adv_success += 1
            
            # 画图 (只画前 3 张)
            if 2 <= adv_success <= 4:
                print(f"\n攻击成功！真值: {classes[target.item()]} -> 攻击后: {classes[final_pred.item()]}")
                
                # 准备显示用的 numpy 数据
                img_orig = data_raw[0].cpu().numpy().transpose(1, 2, 0)
                img_adv = perturbed_data_raw[0].cpu().numpy().transpose(1, 2, 0)
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title(f"Original: {classes[target.item()]}")
                plt.imshow(img_orig)
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.title(f"PGD Attack: {classes[final_pred.item()]}")
                plt.imshow(img_adv)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'pgd_example_{adv_success}.png') # 保存而不是 show，防止卡住
                # plt.show() 

    pbar.close()
    print(f"\n攻击完成！样本数: {total}")
    print(f"攻击成功数 (Adv Success): {adv_success}")
    print(f"攻击成功率 (Error Rate): {adv_success/total:.2%}")
    print(f"模型鲁棒准确率 (Robust Acc): {1 - adv_success/total:.2%}")


if __name__ == '__main__':
    main()