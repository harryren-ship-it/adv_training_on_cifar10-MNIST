import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 模型定义 
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
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

# ==========================================
# 2. 生成随机噪声 (Random Noise)
# ==========================================
'''def generate_batch_random_noise(data, epsilon):
    """
    生成随机噪声样本 (Control Experiment)
    方法1：randomly adding ± epsilon to each pixel
    """
    # 生成和 data 形状一样的高斯噪声，然后取符号，这就变成了 +1 或 -1
    # 乘以 epsilon，就变成了 +0.25 或 -0.25
    noise = torch.randn_like(data).sign() * epsilon
    
    # 加上噪声
    perturbed_data = data + noise
    
    # 截断到 [0, 1]
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data'''

def generate_batch_random_noise(data, epsilon):
    """
    方法2：生成均匀分布噪声 U(-epsilon, epsilon)
    """
    # 1. 生成 [0, 1) 的随机数
    noise = torch.rand_like(data)
    
    # 2. 变换区间到 [-epsilon, epsilon)
    # 公式：[0, 1] * 2*eps - eps  ==>  [-eps, eps]
    noise = noise * 2 * epsilon - epsilon
    
    # 3. 加上噪声
    perturbed_data = data + noise
    
    # 4. 截断到 [0, 1] (必须做，否则像素值会溢出)
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

# ==========================================
# 3. 噪声训练循环 (Noise Training Loop)
# ==========================================
def train_noise_augmentation(model, device, train_loader, optimizer, epoch, epsilon=0.25):
    model.train()
    
    total_clean_loss = 0
    total_noise_loss = 0
    correct_clean = 0
    correct_noise = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # --- 生成的是随机噪声，不是对抗样本 ---
        x_noise = generate_batch_random_noise(data, epsilon)

        optimizer.zero_grad()

        # 1. 干净 Loss
        output_clean = model(data)
        loss_clean = F.cross_entropy(output_clean, target)
        
        # 2. 噪声 Loss
        output_noise = model(x_noise)
        loss_noise = F.cross_entropy(output_noise, target)

        # 3. 混合 Loss
        total_loss = 0.5 * loss_clean + 0.5 * loss_noise

        total_loss.backward()
        optimizer.step()

        # --- 统计 ---
        batch_size = len(data)
        total_samples += batch_size
        total_clean_loss += loss_clean.item() * batch_size
        total_noise_loss += loss_noise.item() * batch_size
        
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        correct_clean += pred_clean.eq(target.view_as(pred_clean)).sum().item()
        
        pred_noise = output_noise.argmax(dim=1, keepdim=True)
        correct_noise += pred_noise.eq(target.view_as(pred_noise)).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\t'
                  f'Loss(Clean): {total_clean_loss/total_samples:.4f}  Loss(Noise): {total_noise_loss/total_samples:.4f}\t'
                  f'Acc(Clean): {100.*correct_clean/total_samples:.1f}%  Acc(Noise): {100.*correct_noise/total_samples:.1f}%')

# ==========================================
# 4. 验证函数：用 FGSM 攻击来测试它！
# ==========================================
def eval_robustness_against_fgsm(model, device, test_loader, epsilon=0.25):
    model.eval()
    correct_clean = 0
    correct_fgsm = 0 # 真正的 FGSM 攻击准确率
    
    print("\n正在进行真正的 FGSM 攻击测试 (Exam)...")
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 1. 干净准确率
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct_clean += pred.eq(target.view_as(pred)).sum().item()

        # 2. 真正的 FGSM 攻击 (注意：这里要算梯度！)
        data.requires_grad = True
        output = model(data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        # 生成 FGSM 样本
        x_adv = data + epsilon * data_grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # 测试
        output_adv = model(x_adv)
        pred_adv = output_adv.argmax(dim=1, keepdim=True)
        correct_fgsm += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    acc_clean = 100. * correct_clean / len(test_loader.dataset)
    acc_fgsm = 100. * correct_fgsm / len(test_loader.dataset)
    
    print(f'Test set: Clean Accuracy: {acc_clean:.2f}%')
    print(f'Test set: FGSM Accuracy (Epsilon={epsilon}): {acc_fgsm:.2f}%')
    print("-----------------------------------------------------------")

# ==========================================
# 5. 主程序
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        datasets.MNIST('./raw/data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST('./raw/data', train=False, download=False, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False)

    # 初始化全新的模型
    print("初始化全新的模型 (用于随机噪声训练)...")
    model_noise = MnistModel().to(device)
    optimizer = optim.Adam(model_noise.parameters(), lr=0.001)

    EPOCHS = 5
    EPSILON = 0.25
    
    print(f"开始随机噪声训练 (Control Experiment)... Epsilon = {EPSILON}")
    
    for epoch in range(1, EPOCHS + 1):
        train_noise_augmentation(model_noise, device, train_loader, optimizer, epoch, epsilon=EPSILON)
        
        # 每一轮结束后，用 FGSM 攻击
        eval_robustness_against_fgsm(model_noise, device, test_loader, epsilon=EPSILON)

    torch.save(model_noise.state_dict(), "raw/model/mnist_random_noise_trained.pkl")
    print("模型已保存: mnist_random_noise_trained.pkl")

if __name__ == '__main__':
    main()