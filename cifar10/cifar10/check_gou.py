import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("GPU 设备数量:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("当前显卡:", torch.cuda.get_device_name(0))
else:
    print("❌ 悲剧了，PyTorch 没找到显卡，正在使用 CPU。")