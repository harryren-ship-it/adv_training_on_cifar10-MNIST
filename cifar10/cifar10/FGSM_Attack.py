import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.readData import read_dataset
from utils.ResNet import ResNet18

# --------------------------
# 配置区域
# --------------------------
EPSILON = 0.1  # 论文中的 epsilon
BATCH_SIZE = 1 # 为了方便可视化，我们将Batch设为1，一张张攻击
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'cifar10/checkpoint/resnet18_cifar10_trades_best.pt'

# 定义反归一化，用于显示图片
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def denormalize(tensor):
    """把 PyTorch 的 Tensor 图片转回人类能看的 numpy 数组"""
    img = tensor.clone().detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

# --------------------------
# 核心：FGSM 攻击函数
# --------------------------
def fgsm_attack(image, epsilon, data_grad):
    # 1. 获取梯度的符号 (Sign)
    sign_data_grad = data_grad.sign()
    
    # 2. 生成扰动 (Noise)
    # 公式：eta = epsilon * sign(grad)
    perturbed_image = image + epsilon * sign_data_grad
    
    # 3. 如果为了严谨，可以加上裁剪，确保像素数值不越界（可选）
    # 但在Normalized空间下，边界比较模糊，这里为了还原论文效果直接加
    return perturbed_image

def main():
    print(f"正在复现 FGSM 攻击，Epsilon = {EPSILON}...")

    # 1. 加载数据
    _, _, test_loader = read_dataset(batch_size=BATCH_SIZE, pic_path='cifar10/dataset')

    # 2. 加载模型
    model = ResNet18()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, 10)
    
    # 加载权重
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval() # 极其重要！固定 BatchNorm 和 Dropout

    # 3. 统计变量
    correct = 0            # 原始图片预测正确的数量
    adv_success = 0        # 攻击成功的数量（原本对的，攻击后错了）
    total = 0
    wrong_confidences = [] # 记录攻击后预测错误的置信度

    # 4. 开始循环攻击
    print("开始攻击测试集 (这可能需要几分钟)...")
    
    # 这里的循环用于统计数据
    # enumerate(test_loader) 会稍微慢一点，因为 batch_size=1
    for i, (data, target) in enumerate(test_loader):
        if total >= 1000: break # 为了快速看结果，先只跑1000张

        data, target = data.to(DEVICE), target.to(DEVICE)

        # 设置 requires_grad=True
        # 因为我们要对“输入图片”求导，而不是对模型参数求导
        data.requires_grad = True

        # 前向传播
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # 拿到原始预测结果

        # 如果一开始就预测错了，那就不需要攻击了（或者不算在攻击成功率里）
        if init_pred.item() != target.item():
            continue

        # 计算 Loss
        loss = F.cross_entropy(output, target)

        # 反向传播，计算 Input 的梯度
        model.zero_grad()
        loss.backward()
        
        # 拿到图片的梯度
        data_grad = data.grad.data

        # 实施 FGSM 攻击
        perturbed_data = fgsm_attack(data, EPSILON, data_grad)

        # 再次分类（预测对抗样本）
        final_output = model(perturbed_data)
        final_pred = final_output.max(1, keepdim=True)[1] # 拿到攻击后的预测

        # 统计
        total += 1
        
        # 检查是否攻击成功
        if final_pred.item() != target.item():
            adv_success += 1
            # 计算攻击后的置信度 (Probability)
            probs = F.softmax(final_output, dim=1)
            wrong_conf_val = probs.max().item()
            wrong_confidences.append(wrong_conf_val)

            # --- 可视化：抽取第1张成功的案例画出来 ---
            if adv_success == 5:
                plot_adv_example(data, perturbed_data, epsilon=EPSILON, 
                                 orig_pred=classes[init_pred.item()], 
                                 adv_pred=classes[final_pred.item()])

    # 5. 输出论文风格的结论
    acc = 100. * (total - adv_success) / total
    error_rate = 100. * adv_success / total # 攻击成功率
    avg_conf = np.mean(wrong_confidences) * 100 if len(wrong_confidences) > 0 else 0

    print("\n" + "="*30)
    print(f"FGSM Attack Result (Epsilon = {EPSILON})")
    print("="*30)
    print(f"Total Evaluated Images (Correctly Classified Originally): {total}")
    print(f"Adversarial Error Rate (Attack Success Rate): {error_rate:.2f}%")
    print(f"Model Accuracy after Attack: {acc:.2f}%")
    print(f"Avg. Confidence on Wrong Labels: {avg_conf:.2f}%")
    print("="*30)
    print(f"论文复现对比: 论文中 Error Rate 87.15%, Avg Conf 96.6%")
    print(f"注意: 由于模型架构不同(ResNet vs Maxout)，数值会有差异，但现象应一致。")

def plot_adv_example(orig_tensor, adv_tensor, epsilon, orig_pred, adv_pred):
    """画出那张经典的熊猫图风格对比"""
    noise = adv_tensor - orig_tensor
    
    orig_img = denormalize(orig_tensor[0])
    adv_img = denormalize(adv_tensor[0])
    noise_img = noise[0].clone().detach().cpu().numpy().transpose(1, 2, 0)
    
    # 放大噪声以便观看，否则肉眼可能看不见
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original: {orig_pred}")
    plt.imshow(orig_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Noise (x {epsilon})")
    plt.imshow(noise_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Adversarial: {adv_pred}")
    plt.imshow(adv_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()