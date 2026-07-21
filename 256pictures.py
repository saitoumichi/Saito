import torch
import numpy as np

# x_train から1枚だけ取り出す
x = x_train[0:1]  # shape: (1, 256, 256, 256) の想定

# PyTorch形式に変換: (B, D, H, W) → (B, C, D, H, W)
x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)

print("元画像:", x.shape)

# Haarウェーブレット変換
w = haar_wavelet_3d(x)
print("Wavelet後:", w.shape)

# 逆Haarウェーブレット変換
x_rec = inverse_haar_wavelet_3d(w)
print("逆変換後:", x_rec.shape)

# 誤差確認
mean_error = torch.mean(torch.abs(x - x_rec)).item()
max_error = torch.max(torch.abs(x - x_rec)).item()

print("mean error:", mean_error)
print("max error:", max_error)