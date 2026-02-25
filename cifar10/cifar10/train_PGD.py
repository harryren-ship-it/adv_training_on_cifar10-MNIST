import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.readData import read_dataset
from utils.ResNet import ResNet18

# --------------------------
# 1. 配置与超参数
# --------------------------
EPSILON = 8/255  # 攻击扰动范围
ALPHA_TRAIN = 2.5/255  # 训练步长
ALPHA_TEST = 2/255 # 测试步长
ITERS_TRAIN = 7  # 训练时迭代次数（平衡速度与质量）
ITERS_TEST = 10  # 测试时迭代次数
BATCH_SIZE = 128 # 训练时用较大的 Batch Size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLEAN_CHECKPOINT = 'cifar10/checkpoint/resnet18_cifar10.pt'
ADV_CHECKPOINT = 'cifar10/checkpoint/resnet18_cifar10_pgd_best.pt'

mean_list = [0.485, 0.456, 0.406]
std_list = [0.229, 0.224, 0.225]

# --------------------------
# 2. 归一化辅助函数
# --------------------------
def normalize_tensor(batch_img):
    mean = torch.tensor(mean_list).view(1, 3, 1, 1).to(batch_img.device)
    std = torch.tensor(std_list).view(1, 3, 1, 1).to(batch_img.device)
    return (batch_img - mean) / std

def unnormalize_tensor(batch_img):
    mean = torch.tensor(mean_list).view(1, 3, 1, 1).to(batch_img.device)
    std = torch.tensor(std_list).view(1, 3, 1, 1).to(batch_img.device)
    return batch_img * std + mean

# --------------------------
# 3. 核心攻击函数
# --------------------------
def pgd_attack(model, images_raw, labels, epsilon, alpha, iters):
    """ 基于 [0, 1] 原始图像生成 PGD 对抗样本 """
    model.eval() # 生成样本时固定 BatchNorm
    ori_images = images_raw.clone().detach()
    
    # 随机启动
    adv_images = images_raw.clone().detach() + torch.empty_like(images_raw).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1).detach().requires_grad_(True)
    
    for _ in range(iters):
        # 归一化后输入模型
        outputs = model(normalize_tensor(adv_images))
        loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = adv_images.grad.detach()
        # 更新并投影
        adv_images = adv_images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, -epsilon, epsilon)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach().requires_grad_(True)
        
    return adv_images.detach()

def fgsm_attack(model, images_raw, labels, epsilon):
    """ FGSM 攻击用于评估 """
    model.eval()
    images_raw.requires_grad = True
    outputs = model(normalize_tensor(images_raw))
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images_raw + epsilon * images_raw.grad.sign()
    return torch.clamp(adv_images, 0, 1).detach()

# --------------------------
# 4. 对抗训练逻辑 
# --------------------------
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    loop = tqdm(loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        # 注意：loader 出来的 images 已经是归一化过的，需要先还原
        images_raw = unnormalize_tensor(images)
        
        # -----------------------------------------------------------
        # 步骤 1: 生成对抗样本 (基于当前时刻的模型参数)
        # -----------------------------------------------------------
        # 在生成对抗样本时，通常建议将模型设为 eval，防止 BN 层抖动
        model.eval()
        adv_images_raw = pgd_attack(model, images_raw, labels, EPSILON, ALPHA_TRAIN, ITERS_TRAIN)
        model.train() # 切回训练模式准备更新参数

        # -----------------------------------------------------------
        # 步骤 2: 第一次更新 - 使用标准样本 (Clean Loss)
        # -----------------------------------------------------------
        optimizer.zero_grad()
        out_clean = model(normalize_tensor(images_raw))
        loss_clean = F.cross_entropy(out_clean, labels)
        loss_clean.backward()
        optimizer.step() # <--- 第一次参数更新
        
        # -----------------------------------------------------------
        # 步骤 3: 第二次更新 - 使用对抗样本 (Adv Loss)
        # 注意：由于上一步参数已经变了，这里必须重新跑一遍前向传播 (Forward)
        # -----------------------------------------------------------
        optimizer.zero_grad()
        out_adv = model(normalize_tensor(adv_images_raw))
        loss_adv = F.cross_entropy(out_adv, labels)
        loss_adv.backward()
        optimizer.step() # <--- 第二次参数更新
        
        # -----------------------------------------------------------
        # 统计
        # -----------------------------------------------------------
        total += labels.size(0)
        with torch.no_grad():
            correct_clean += (out_clean.argmax(1) == labels).sum().item()
            correct_adv += (out_adv.argmax(1) == labels).sum().item()
        
        loop.set_description(f"Epoch [{epoch}]")
        # 显示两个 Loss 的均值或当前的 Adv Loss
        loop.set_postfix(ClnLoss=f"{loss_clean.item():.3f}", 
                         AdvLoss=f"{loss_adv.item():.3f}", 
                         ClnAcc=f"{100.*correct_clean/total:.1f}%", 
                         AdvAcc=f"{100.*correct_adv/total:.1f}%")

# --------------------------
# 5. 综合评估函数
# --------------------------
def evaluate(model, loader, device):
    model.eval()
    c_clean, c_fgsm, c_pgd, total = 0, 0, 0, 0
    
    print("\n正在进行标准样本、FGSM、PGD 三重测试...")
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        images_raw = unnormalize_tensor(images)
        total += labels.size(0)
        
        with torch.no_grad():
            # 标准测试
            c_clean += (model(normalize_tensor(images_raw)).argmax(1) == labels).sum().item()
        
        # 对抗测试 (需要梯度) 
        with torch.enable_grad():
            adv_fgsm = fgsm_attack(model, images_raw.clone(), labels, EPSILON)
            adv_pgd = pgd_attack(model, images_raw.clone(), labels, EPSILON, ALPHA_TEST, ITERS_TEST)
            
        c_fgsm += (model(normalize_tensor(adv_fgsm)).argmax(1) == labels).sum().item()
        c_pgd += (model(normalize_tensor(adv_pgd)).argmax(1) == labels).sum().item()
        
    print(f"测试完成 -> Clean Acc: {100.*c_clean/total:.2f}% | FGSM Acc: {100.*c_fgsm/total:.2f}% | PGD Acc: {100.*c_pgd/total:.2f}%")
    return 100.*c_pgd/total

# --------------------------
# 6. 主函数
# --------------------------
def main():
    # 数据加载
    train_loader, _, test_loader = read_dataset(batch_size=BATCH_SIZE, pic_path='cifar10\\dataset')
    
    # 初始化 ResNet18 
    model = ResNet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)
    
    # 尝试加载预训练模型
    try:
        model.load_state_dict(torch.load(CLEAN_CHECKPOINT, map_location=DEVICE))
        print("已加载预训练 CIFAR-10 模型，开始对抗微调。")
    except:
        print("未找到模型，将从头开始对抗训练。")
        
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    
    best_robust_acc = 0.0
    EPOCHS = 20
    
    for epoch in range(1, EPOCHS + 1  ):
        train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        
        # 每轮测试一次
        robust_acc = evaluate(model, test_loader, DEVICE)
        
        # 保存最佳鲁棒模型
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            print(f"发现最佳鲁棒模型 (PGD Acc: {best_robust_acc:.2f}%)，正在保存...")
            torch.save(model.state_dict(), ADV_CHECKPOINT)

if __name__ == '__main__':
    main()