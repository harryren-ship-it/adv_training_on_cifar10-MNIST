import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. 模型定义 (支持拆分 Forward)
# ==========================================
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        
        # 观察 Linear1，因为它是注入噪声的地方
        self.linear1 = nn.Linear(320, 128) 
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward_part1(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x) # <--- 这里的输出值
        return x

    def forward_part2(self, h):
        x = self.linear2(h)
        x = self.linear3(x)
        return x

    def forward(self, x):
        h = self.forward_part1(x)
        out = self.forward_part2(h)
        return out

# ==========================================
# 2. 两个辅助函数：生成不同位置的噪声
# ==========================================
# A. 针对输入层 (Input FGSM)
def generate_input_adv(model, x, target, epsilon):
    x_copy = x.detach().clone()
    x_copy.requires_grad = True
    out = model(x_copy)
    loss = F.cross_entropy(out, target)
    loss.backward()
    return torch.clamp(x + epsilon * x_copy.grad.sign(), 0, 1).detach()

# B. 针对隐藏层 (Hidden FGSM)
def generate_hidden_adv(model, x, target, epsilon):
    h = model.forward_part1(x)
    h = h.detach().clone()
    h.requires_grad = True
    out = model.forward_part2(h)
    loss = F.cross_entropy(out, target)
    loss.backward()
    # 注意：隐藏层没有 clamp，允许无限变大
    return (h + epsilon * h.grad.sign()).detach()

# ==========================================
# 3. 训练函数
# ==========================================
def train(model, device, loader, optimizer, mode='input', epsilon=0.25):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        if mode == 'input':
            # 输入层对抗训练
            x_adv = generate_input_adv(model, data, target, epsilon)
            optimizer.zero_grad()
            loss = 0.5 * F.cross_entropy(model(data), target) + \
                   0.5 * F.cross_entropy(model(x_adv), target)
            loss.backward()
            optimizer.step()
            
        elif mode == 'hidden':
            # 隐藏层对抗训练
            h_adv = generate_hidden_adv(model, data, target, epsilon)
            optimizer.zero_grad()
            # 必须拆开跑
            h_clean = model.forward_part1(data)
            out_clean = model.forward_part2(h_clean)
            out_adv = model.forward_part2(h_adv)
            
            loss = 0.5 * F.cross_entropy(out_clean, target) + \
                   0.5 * F.cross_entropy(out_adv, target)
            loss.backward()
            optimizer.step()

# ==========================================
# 4. 主程序：对比实验
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(datasets.MNIST('./raw/data', train=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    
    # --- 实验组 1: 输入层对抗模型 (Reference) ---
    print("1. 正在训练参照组：Input Adversarial Model ...")
    model_input = MnistModel().to(device)
    opt_input = optim.Adam(model_input.parameters(), lr=0.001)
    for epoch in range(2): # 训练轮次少一点，快速验证
        train(model_input, device, train_loader, opt_input, mode='input')
        
    # --- 实验组 2: 隐藏层对抗模型 (Experiment) ---
    print("2. 正在训练大嗓门组：Hidden Adversarial Model ...")
    model_hidden = MnistModel().to(device)
    opt_hidden = optim.Adam(model_hidden.parameters(), lr=0.001)
    for epoch in range(3):
        train(model_hidden, device, train_loader, opt_hidden, mode='hidden')

    # ==========================================
    # 5. 测量激活值和权重
    # ==========================================
    print("\n========== 结果对比分析 ==========")
    
    # A. 测量权重 (Linear1 的 Weight L2 Norm)
    # 看看参数本身是不是变大了
    w_norm_input = model_input.linear1.weight.norm().item()
    w_norm_hidden = model_hidden.linear1.weight.norm().item()
    
    print(f"[权重大小对比] Linear1 Weight Norm:")
    print(f"  - Input Model : {w_norm_input:.2f}")
    print(f"  - Hidden Model: {w_norm_hidden:.2f}")
    print(f"  - 倍数关系    : {w_norm_hidden / w_norm_input:.2f}x")
    
    # B. 测量激活值 (Activation)
    # 跑一批干净数据，看看中间层输出的数值有多大
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)
    
    with torch.no_grad():
        h_input = model_input.forward_part1(test_batch)
        h_hidden = model_hidden.forward_part1(test_batch)
        
    # 计算平均绝对值 (Mean Absolute Value) 和 最大值
    mean_act_input = h_input.abs().mean().item()
    max_act_input = h_input.abs().max().item()
    
    mean_act_hidden = h_hidden.abs().mean().item()
    max_act_hidden = h_hidden.abs().max().item()
    
    print(f"\n[激活值大小对比] Linear1 Output (Activation):")
    print(f"  - Input Model : Mean={mean_act_input:.2f}, Max={max_act_input:.2f}")
    print(f"  - Hidden Model: Mean={mean_act_hidden:.2f}, Max={max_act_hidden:.2f}")
    print(f"  - Mean 倍数   : {mean_act_hidden / mean_act_input:.2f}x")
    
    if mean_act_hidden > mean_act_input * 1.5:
        print("\n[结论] 验证成功！隐藏层对抗训练导致模型单纯通过放大数值来抵御噪声。")
    else:
        print("\n[结论] 差异不明显，可能需要训练更多轮次。")

if __name__ == '__main__':
    main()