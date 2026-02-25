import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPSILON = 0.3
ITERS = 20
ALPHA = (EPSILON * 1.5) / ITERS  # 保证抵达边缘
EPOCHS = 10     
LR = 0.0005

# ==========================================
# 1. MnistModel 类定义
# ==========================================
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

# ==========================================
# 2. 核心：现场生成对抗样本的函数
# ==========================================
def pgd_attack(model, data, target, epsilon, alpha, iters):
    """
    在训练循环中调用的生成函数
    """
    # 1. 保护原始数据，复制一份，并允许求导
    data_copy = data.clone().detach().requires_grad_(True)
    
    #随机初始化
    adv_images = data.clone().detach() + torch.empty_like(data).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)
    
    for i in range(iters):
        adv_images = adv_images.detach().requires_grad_(True)
        
        output = model(adv_images)
        loss = F.cross_entropy(output, target)
        
        model.zero_grad()
        loss.backward()
        
        # 获取当前对抗样本的梯度
        data_grad = adv_images.grad.data
        
        # PGD 更新：x_adv = x_adv + alpha * sign(grad)
        adv_images = adv_images + alpha * data_grad.sign()
        
        # 投影回 [0, 1] 范围内，并且确保在 epsilon 范围内
        adv_images = torch.max(torch.min(adv_images, data + epsilon), data - epsilon)
        adv_images = torch.clamp(adv_images, 0, 1)
        
    return adv_images.detach() # 带上detach，不需要对生成过程本身求导



# ==========================================
# 3. 对抗训练循环 (Adversarial Training Loop)
# ==========================================
def train_adversarial(model, device, train_loader, optimizer, epoch, epsilon):
    
    model.train()
    
    # 累加器
    total_clean_loss = 0
    total_adv_loss = 0
    correct_clean = 0
    correct_adv = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 1. 生成对抗样本
        x_adv = pgd_attack(model, data, target, epsilon, alpha = ALPHA, iters = ITERS)

        # 2. 清零梯度
        optimizer.zero_grad()

        # 3. 分别计算 Loss
        output_clean = model(data)
        loss_clean = F.cross_entropy(output_clean, target)
        
        output_adv = model(x_adv)
        loss_adv = F.cross_entropy(output_adv, target)

        # 4. 混合 Loss (Alpha = 0.5)
        total_loss = 0.5 * loss_clean + 0.5 * loss_adv

        # 5. 反向传播
        total_loss.backward()
        optimizer.step()

        # --- 统计数据 ---
        batch_size = len(data)
        total_samples += batch_size
        
        # 累加 Loss 用于显示 (平滑显示)
        total_clean_loss += loss_clean.item() * batch_size
        total_adv_loss += loss_adv.item() * batch_size
        
        # 统计准确率
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        correct_clean += pred_clean.eq(target.view_as(pred_clean)).sum().item()
        
        pred_adv = output_adv.argmax(dim=1, keepdim=True)
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

        if batch_idx % 100 == 0:
            # 计算当前的平均值
            avg_clean_loss = total_clean_loss / total_samples
            avg_adv_loss = total_adv_loss / total_samples
            acc_clean = 100. * correct_clean / total_samples
            acc_adv = 100. * correct_adv / total_samples
            
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\t'
                  f'ClnLoss: {avg_clean_loss:.4f}  AdvLoss: {avg_adv_loss:.4f}\t' # <--- 分开显示了
                  f'ClnAcc: {acc_clean:.1f}%  AdvAcc: {acc_adv:.1f}%')

def fgsm_attack(model, data, target, epsilon):
    """FGSM 攻击用于测试对比"""
    data_copy = data.clone().detach().requires_grad_(True)
    output = model(data_copy)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    perturbed_data = data + epsilon * data_copy.grad.data.sign()
    return torch.clamp(perturbed_data, 0, 1).detach()


# ==========================================
# 4. 综合评估：标准、FGSM、PGD 三重测试
# ==========================================
def eval_all(model, device, test_loader, epsilon):
    model.eval()
    correct_clean = 0
    correct_fgsm = 0
    correct_pgd = 0
    
    print("\n[Evaluation] Testing on Clean, FGSM, and PGD samples...")
    
    with torch.no_grad(): 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 1. Clean
            out_clean = model(data)
            correct_clean += out_clean.argmax(1).eq(target).sum().item()
            
            # 2. FGSM (需开启求导生成)
            with torch.enable_grad():
                adv_fgsm = fgsm_attack(model, data, target, epsilon)
            out_fgsm = model(adv_fgsm)
            correct_fgsm += out_fgsm.argmax(1).eq(target).sum().item()
            
            # 3. PGD (需开启求导生成)
            with torch.enable_grad():
                adv_pgd = pgd_attack(model, data, target, epsilon, alpha=ALPHA, iters=ITERS)
            out_pgd = model(adv_pgd)
            correct_pgd += out_pgd.argmax(1).eq(target).sum().item()

    total = len(test_loader.dataset)
    clean_acc = 100. * correct_clean / total
    fgsm_acc = 100. * correct_fgsm / total
    pgd_acc = 100. * correct_pgd / total
    
    print(f'Results - Clean: {clean_acc:.2f}% | FGSM: {fgsm_acc:.2f}% | PGD: {pgd_acc:.2f}%')
    return pgd_acc # 返回 PGD 准确率作为保存依据

# ==========================================
# 5.主函数
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        datasets.MNIST('./raw/data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST('./raw/data', train=False, download=False, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False)

    # 1. 先初始化结构
    model = MnistModel().to(device)
    
    # 2. 加载之前训练好的参数 
    print("正在加载预训练模型...")
    try:
        state_dict = torch.load("raw/model/model.pkl", map_location=device)
        model.load_state_dict(state_dict)
        print("预训练参数加载成功！在此基础上进行对抗微调。")
    except Exception as e:
        print(f"加载失败，将从头开始训练。错误: {e}")

    # 3. 定义优化器 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_adv_acc = 0.0 # 记录历史最佳

    print(f"开始对抗训练... Epsilon = {EPSILON}，Alpha = {ALPHA:.4f}，迭代次数 = {ITERS}")
    
    for epoch in range(1, EPOCHS + 1):
        train_adversarial(model, device, train_loader, optimizer, epoch, epsilon=EPSILON)
        
        # 获取当前的对抗准确率
        current_adv_acc = eval_all(model, device, test_loader, epsilon=EPSILON)
        
        # --- Save Best 策略 ---
        if current_adv_acc > best_adv_acc:
            print(f"==> 发现新纪录！Adv Acc 从 {best_adv_acc:.2f}% 提升到了 {current_adv_acc:.2f}%。保存模型...")
            best_adv_acc = current_adv_acc
            torch.save(model.state_dict(), "raw/model/mnist_pgd_trained_best.pkl")
            patience = 0
        else:
            print(f"==> 本轮 Adv Acc ({current_adv_acc:.2f}%) 未超过历史最佳 ({best_adv_acc:.2f}%)。")
            patience += 1
            

    print(f"训练结束！最佳对抗准确率: {best_adv_acc:.2f}%")

if __name__ == '__main__':
    main()