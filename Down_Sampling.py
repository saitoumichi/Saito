import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Analysis_Filter.py で保存した w を読み込む
w_path = r"C:\Users\ri0151fv\Saito\analysis_output.npy"
w_np = np.load(w_path)

w = torch.tensor(w_np, dtype=torch.float32).to(device)

print("Analysis後:", w.shape)

# ダウンサンプリング：2個に1個だけ残す
w_down = w[:, :, ::2, ::2, ::2]

print("Downsampling後:", w_down.shape)

# 保存
save_path = r"C:\Users\ri0151fv\Saito\analysis_downsampled.npy"
np.save(save_path, w_down.detach().cpu().numpy())
print("保存先:", save_path)

# 表示
names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

slice_down = w_down.shape[2] // 2

plt.figure(figsize=(16, 8))

for i, name in enumerate(names):
    plt.subplot(2, 4, i + 1)
    plt.imshow(w_down[0, i, slice_down].detach().cpu().numpy(), cmap="gray")
    plt.title(name + " downsampled")
    plt.axis("off")

save_fig_path = r"C:\Users\ri0151fv\Saito\downsampled_8bands.png"
plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
print("画像保存先:", save_fig_path)

plt.show()