import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
def generate_batch_adversarial(model, data, target, epsilon):
    """
    在训练循环中调用的生成函数
    """
    # 1. 保护原始数据，复制一份，并允许求导
    data_copy = data.clone().detach().requires_grad_(True)
    
    # 2. 跑一次前向传播
    output = model(data_copy)
    loss = F.cross_entropy(output, target)
    
    # 3. 反向传播，只为了求输入的梯度 (data.grad)
    model.zero_grad()
    loss.backward()
    data_grad = data_copy.grad.data
    
    # 4. 生成对抗样本 (FGSM)
    # x_adv = x + epsilon * sign(grad)
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    
    # 5. 截断到 [0, 1] 范围
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data.detach() # 记得 detach，不需要对生成过程本身求导


# ==========================================
# 3. 对抗训练循环 (Adversarial Training Loop)
# ==========================================
def train_adversarial(model, device, train_loader, optimizer, epoch, epsilon=0.25):
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
        x_adv = generate_batch_adversarial(model, data, target, epsilon)

        # 2. 清零梯度
        optimizer.zero_grad()

        # 3. 分别计算 Loss
        output_clean = model(data)
        loss_clean = F.cross_entropy(output_clean, target)
        
        output_adv = model(x_adv)
        loss_adv = F.cross_entropy(output_adv, target)

        # 4. 混合 Loss (Alpha = 0.5)
        total_loss = 0.6 * loss_clean + 0.4 * loss_adv

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

# ==========================================
# 修改后的验证函数：增加返回值，以便主函数判断
# ==========================================
def eval_robustness(model, device, test_loader, epsilon=0.25):
    model.eval()
    correct = 0
    correct_adv = 0
    
    print("\nEvaluating Robustness...")
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 干净准确率
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # 对抗准确率
        data.requires_grad = True
        out_temp = model(data)
        loss = F.cross_entropy(out_temp, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        x_adv_test = data + epsilon * data_grad.sign()
        x_adv_test = torch.clamp(x_adv_test, 0, 1)
        
        output_adv = model(x_adv_test)
        pred_adv = output_adv.argmax(dim=1, keepdim=True)
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    clean_acc = 100. * correct / len(test_loader.dataset)
    adv_acc = 100. * correct_adv / len(test_loader.dataset)
    
    print(f'Test set: Clean Accuracy: {clean_acc:.2f}%')
    print(f'Test set: Adversarial Accuracy (Epsilon={epsilon}): {adv_acc:.2f}%\n')
    
    # 返回 Adv Acc 供主函数做 Early Stopping 判断
    return adv_acc

# ==========================================
# 主函数：加入 Save Best 策略
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
    
    print("正在加载预训练模型...")
    try:
        state_dict = torch.load("raw/model/model.pkl", map_location=device)
        model.load_state_dict(state_dict)
        print("预训练参数加载成功！在此基础上进行对抗微调。")
    except Exception as e:
        print(f"加载失败，将从头开始训练。错误: {e}")

    # 3. 定义优化器 (继续往下走)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10     
    EPSILON = 0.25
    
    best_adv_acc = 0.0 # 记录历史最佳

    print(f"开始对抗训练... Epsilon = {EPSILON}")
    
    for epoch in range(1, EPOCHS + 1):
        train_adversarial(model, device, train_loader, optimizer, epoch, epsilon=EPSILON)
        
        # 获取当前的对抗准确率
        current_adv_acc = eval_robustness(model, device, test_loader, epsilon=EPSILON)
        
        # --- Save Best 策略 ---
        if current_adv_acc > best_adv_acc:
            best_adv_acc = current_adv_acc
            print(f"==> 发现新纪录！Adv Acc 从 {best_adv_acc:.2f}% 提升到了 {current_adv_acc:.2f}%。保存模型...")
            torch.save(model.state_dict(), "raw/model/mnist_adv_trained_best.pkl")
            patience = 0
        else:
            print(f"==> 本轮 Adv Acc ({current_adv_acc:.2f}%) 未超过历史最佳 ({best_adv_acc:.2f}%)。")
            patience += 1
            

    print(f"训练结束！最佳对抗准确率: {best_adv_acc:.2f}%")

if __name__ == '__main__':
    main()