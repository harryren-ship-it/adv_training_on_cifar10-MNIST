import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ================================================================
# 1. 模型类定义 
# ================================================================
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

# ================================================================
# 2. FGSM 攻击函数
# ================================================================
def fgsm_attack(image, epsilon, data_grad):
    # 收集梯度的符号
    sign_data_grad = data_grad.sign()
    # 创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 裁剪范围 [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# ================================================================
# 3. 测试与攻击流程
# ================================================================
def test_attack(model, device, test_loader, epsilon):
    correct = 0
    adv_success = 0 # 记录原本分对，但被攻击后分错的样本
    total_clean_correct = 0 # 记录原本能分对的样本数
    
    model.eval()

    # 循环测试集
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True # 关键

        # 1. 原始前向传播
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # 原始预测

        # 如果原本就分错了，我们不计入对抗攻击成功率
        # 但为了计算整体错误率，我们都算上
        
        # 2. 计算 Loss (用于求梯度)
        loss = F.cross_entropy(output, target)

        # 3. 反向传播求梯度
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # 4. 生成对抗样本
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 5. 对抗样本前向传播
        output_adv = model(perturbed_data)
        final_pred = output_adv.max(1, keepdim=True)[1]

        # 统计逻辑
        # 这里的 Error Rate 指的是：在所有样本中，模型最终预测错误的比例
        batch_correct = final_pred.eq(target.view_as(final_pred)).sum().item()
        correct += batch_correct
        
    final_acc = correct / len(test_loader.dataset)
    error_rate = 1 - final_acc
    
    print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc:.2%}\tError Rate = {error_rate:.2%}")

import matplotlib.pyplot as plt
import numpy as np

def visualize_results(model, device, test_loader, epsilon, num_images=5):
    """
    可视化函数：
    展示 原始图片 vs 噪声 vs 对抗样本
    以及 模型的预测结果
    """
    model.eval()
    
    # 取一个 batch 的数据
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    # 1. 正常的攻击流程
    data.requires_grad = True
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # 原始预测
    
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    
    # 生成对抗样本
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    
    # 对抗样本的预测
    output_adv = model(perturbed_data)
    final_pred = output_adv.max(1, keepdim=True)[1] # 对抗后预测
    
    # --- 开始绘图 ---
    # 我们只画前 num_images 张图
    cnt = 0
    plt.figure(figsize=(10, 10))
    
    for i in range(len(data)):
        if cnt >= num_images:
            break
            
        # 转换数据为 numpy 以便绘图
        # .squeeze() 是为了去掉通道维度 (1, 28, 28) -> (28, 28)
        orig_img = data[i].detach().cpu().numpy().squeeze()
        adv_img = perturbed_data[i].detach().cpu().numpy().squeeze()
        
        # 计算噪声 (对抗样本 - 原始图片)
        noise = adv_img - orig_img
        
        # 获取标签
        true_label = target[i].item()
        orig_label = init_pred[i].item()
        adv_label = final_pred[i].item()
        

        cnt += 1
        
        # 1. 显示原始图片
        plt.subplot(num_images, 3, (cnt-1)*3 + 1)
        plt.title(f"Original: {orig_label} (True: {true_label})", color="green" if orig_label==true_label else "red")
        plt.imshow(orig_img, cmap="gray")
        plt.axis("off")
        
        # 2. 显示噪声 (放大显示以便观察)
        plt.subplot(num_images, 3, (cnt-1)*3 + 2)
        plt.title("Noise (Perturbation)")
        # 这里的噪声是在 [-0.25, 0.25] 之间，为了显示清楚，我们把它平移居中
        # 灰色代表 0，白色代表 +0.25，黑色代表 -0.25
        plt.imshow(noise, cmap="gray", vmin=-epsilon, vmax=epsilon) 
        plt.axis("off")
        
        # 3. 显示对抗样本
        plt.subplot(num_images, 3, (cnt-1)*3 + 3)
        plt.title(f"Adversarial: {adv_label}", color="red" if adv_label!=true_label else "green")
        plt.imshow(adv_img, cmap="gray")
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()



# ================================================================
# 4. 主函数
# ================================================================
def main():
    # 配置
    MODEL_PATH = "raw/model/model.pkl"  
    DATA_ROOT = './raw/data'       
    EPSILON = 0.25                 # 论文中的扰动大小
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    print("正在初始化 CNN 模型...")
    model = MnistModel().to(device)

    # 2. 加载参数 (state_dict)
    print(f"正在加载参数: {MODEL_PATH}")
    try:
        # 加载参数字典
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # 将参数加载进模型
        model.load_state_dict(state_dict)
        print("参数加载成功！")
    except Exception as e:
        print(f"参数加载失败: {e}")
        return

    # 3. 加载数据 (防止重新下载)
    if not os.path.exists(os.path.join(DATA_ROOT, 'MNIST')):
        print(f"错误：请确保 {DATA_ROOT}/MNIST 目录存在")
        return

    test_loader = DataLoader(
        datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=transforms.ToTensor()),
        batch_size=100, shuffle=False)

    # 4. 执行攻击
    print(f"\n开始攻击 (使用 Epsilon = {EPSILON})...")
    
    # 先测一下没有攻击时的准确率 (Epsilon = 0)
    test_attack(model, device, test_loader, epsilon=0)
    
    # 测一下有攻击时的准确率
    test_attack(model, device, test_loader, epsilon=EPSILON)
    
    # 5. 可视化结果
    print("\n正在可视化对抗样本...")
    visualize_results(model, device, test_loader, epsilon=EPSILON, num_images=5)
    
if __name__ == '__main__':
    main()