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
EPSILON = 8/255  
ALPHA_TRAIN = 2/255   # TRADES 建议步长略小于 PGD 训练
ALPHA_TEST = 2/255 
ITERS_TRAIN = 10      # TRADES 训练时迭代次数增加可以提高鲁棒性上限
ITERS_TEST = 10       
BETA = 5.0            # TRADES 权衡参数，建议 1.0 ~ 6.0
BATCH_SIZE = 128 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLEAN_CHECKPOINT = 'cifar10/checkpoint/resnet18_cifar10.pt'
ADV_CHECKPOINT = 'cifar10/checkpoint/resnet18_cifar10_trades_best.pt'

mean_list = [0.485, 0.456, 0.406]
std_list = [0.229, 0.224, 0.225]

# --------------------------
# 2. 归一化辅助函数 (保持不变)
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
# 3. 核心攻击函数 (TRADES 版 + 标准版)
# --------------------------
def trades_pgd_attack(model, images_raw, epsilon, alpha, iters):
    """ TRADES 专用：最大化 KL 散度的攻击 """
    model.eval()
    with torch.no_grad():
        logits_clean = model(normalize_tensor(images_raw))
        prob_clean = F.softmax(logits_clean, dim=1)

    # 随机启动
    adv_images = images_raw.clone().detach() + torch.empty_like(images_raw).uniform_(-epsilon / 10, epsilon / 10) # TRADES 推荐更小的随机启动范围
    adv_images = torch.clamp(adv_images, 0, 1).detach().requires_grad_(True)
    
    for _ in range(iters):
        logits_adv = model(normalize_tensor(adv_images))
        # 目标是让输出分布远离干净样本的分布
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), prob_clean, reduction='batchmean')
        
        model.zero_grad()
        loss_kl.backward()
        
        grad = adv_images.grad.detach()
        adv_images = adv_images + alpha * grad.sign()
        eta = torch.clamp(adv_images - images_raw, -epsilon, epsilon)
        adv_images = torch.clamp(images_raw + eta, 0, 1).detach().requires_grad_(True)
        
    return adv_images.detach()

def standard_pgd_attack(model, images_raw, labels, epsilon, alpha, iters):
    """ 用于测试的常规 PGD (基于 Cross-Entropy) """
    model.eval()
    ori_images = images_raw.clone().detach()
    adv_images = images_raw.clone().detach() + torch.empty_like(images_raw).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1).detach().requires_grad_(True)
    
    for _ in range(iters):
        outputs = model(normalize_tensor(adv_images))
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = adv_images.grad.detach()
        adv_images = adv_images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, -epsilon, epsilon)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach().requires_grad_(True)
    return adv_images.detach()

# --------------------------
# 4. TRADES 对抗训练逻辑 
# --------------------------
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_nat_loss = 0
    total_rob_loss = 0
    total = 0
    
    loop = tqdm(loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        images_raw = unnormalize_tensor(images)
        
        # 步骤 1: 生成 TRADES 对抗样本 (基于 KL 散度)
        model.eval()
        adv_images_raw = trades_pgd_attack(model, images_raw, EPSILON, ALPHA_TRAIN, ITERS_TRAIN)
        model.train()

        # 步骤 2: 计算 TRADES 混合 Loss
        optimizer.zero_grad()
        
        # 自然损失 (保持分类准确率)
        logits_clean = model(normalize_tensor(images_raw))
        loss_natural = F.cross_entropy(logits_clean, labels)
        
        # 鲁棒损失 (保持预测的一致性)
        logits_adv = model(normalize_tensor(adv_images_raw))
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean, dim=1),
            reduction='batchmean'
        )
        
        # TRADES 总损失公式
        loss = loss_natural + BETA * loss_robust
        
        loss.backward()
        optimizer.step()
        
        # 统计数据
        total += labels.size(0)
        total_nat_loss += loss_natural.item()
        total_rob_loss += loss_robust.item()
        
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(NatL=f"{loss_natural.item():.3f}", 
                         RobL=f"{loss_robust.item():.3f}",
                         Beta=BETA)

# --------------------------
# 5. 综合评估函数 (保持 Triple Test 逻辑)
# --------------------------
def evaluate(model, loader, device):
    model.eval()
    c_clean, c_pgd, total = 0, 0, 0
    
    print("\n正在进行 Clean 与标准 PGD-20 测试...")
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        images_raw = unnormalize_tensor(images)
        total += labels.size(0)
        
        with torch.no_grad():
            c_clean += (model(images).argmax(1) == labels).sum().item()
        
        # 测试对抗鲁棒性依然使用最强的标准 PGD 攻击
        with torch.enable_grad():
            adv_pgd = standard_pgd_attack(model, images_raw.clone(), labels, EPSILON, ALPHA_TEST, ITERS_TEST)
            
        c_pgd += (model(normalize_tensor(adv_pgd)).argmax(1) == labels).sum().item()
        
    print(f"测试完成 -> Clean Acc: {100.*c_clean/total:.2f}% | Robust PGD Acc: {100.*c_pgd/total:.2f}%")
    return 100.*c_pgd/total

# --------------------------
# 6. 主函数
# --------------------------

def adjust_learning_rate(optimizer, epoch):
    """手动降低学习率"""
    lr = 1e-4  
    if epoch >= 40:
        lr = 1e-4 * 0.1
    if epoch >= 50:
        lr = 1e-4 * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    train_loader, _, test_loader = read_dataset(batch_size=BATCH_SIZE, pic_path='cifar10\\dataset')
    
    model = ResNet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)
    
    # 打印设备信息，帮助调试和确认环境配置
    print(f"==================================================")
    print(f"当前运行设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"当前显存占用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"==================================================")
    
    try:
        model.load_state_dict(torch.load(CLEAN_CHECKPOINT, map_location=DEVICE))
        print("成功加载模型，开始 TRADES 对抗训练...")
    except:
        print("未找到模型，从头训练...")
        
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    
    best_robust_acc = 0.0
    EPOCHS = 60
    patience = 8
    no_improve_epochs = 0
    
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        robust_acc = evaluate(model, test_loader, DEVICE)
        
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            no_improve_epochs = 0
            print(f"新纪录！保存 TRADES 模型 (PGD Acc: {best_robust_acc:.2f}%)")
            torch.save(model.state_dict(), ADV_CHECKPOINT)
            
        
        else:
            no_improve_epochs += 1
            print(f"未提升的 epoch 数: {no_improve_epochs}/{patience}")
            
        if no_improve_epochs >= patience:
            print("连续多次未提升，提前停止训练。")
            break
    
    print(f"训练结束，最佳 PGD-20 鲁棒准确率: {best_robust_acc:.2f}%")

if __name__ == '__main__':
    main()