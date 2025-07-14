import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import numpy as np

from models.vit import MiniViT
from utils import load_checkpoints

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CIFAR10 类别标签
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = MiniViT(img_size=32, patch_size=4, embed_dim=256, depth=8, heads=8).to(device)
load_checkpoints(model, path="best_model.pt")
model.eval()

# 加载测试集（不打乱）
transform = transforms.ToTensor()
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 随机选择 N 个样本做推理 + 可视化
N = 8
indices = random.sample(range(len(test_dataset)), N)
fig, axes = plt.subplots(1, N, figsize=(3 * N, 3))

for i, idx in enumerate(indices):
    img, label = test_dataset[idx]
    img_input = img.unsqueeze(0).to(device)  # [1, 3, 32, 32]

    with torch.no_grad():
        output = model(img_input)  # [1, 10]
        pred = output.argmax(dim=1).item()
        conf = F.softmax(output, dim=1)[0, pred].item()

    # 可视化图像
    img_np = img.permute(1, 2, 0).numpy()
    axes[i].imshow(img_np)
    axes[i].axis("off")
    axes[i].set_title(
        f"GT: {classes[label]}\nPred: {classes[pred]}\nConf: {conf:.2f}",
        fontsize=10,
        color="green" if pred == label else "red",
    )

plt.tight_layout()
plt.savefig("inference_result.png")
# plt.show()
plt.close()
