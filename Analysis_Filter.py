import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Haar3DAnalysisOnly(nn.Module):
    def __init__(self):
        super().__init__()

        hL = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hH = torch.tensor([1.0, -1.0]) / math.sqrt(2)

        filters = []
        names = []

        for z_name, z_filter in zip(["L", "H"], [hL, hH]):
            for y_name, y_filter in zip(["L", "H"], [hL, hH]):
                for x_name, x_filter in zip(["L", "H"], [hL, hH]):
                    kernel = (
                        z_filter[:, None, None]
                        * y_filter[None, :, None]
                        * x_filter[None, None, :]
                    )
                    filters.append(kernel)
                    names.append(z_name + y_name + x_name)

        weight = torch.stack(filters, dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)
        self.names = names

    def forward(self, x):

        # 各軸の末尾側だけに1ボクセルpadding
        x = F.pad(
           x,
            (0, 1,   # W
             0, 1,   # H
             0, 1)   # D
        )

        # padding済みなのでconv3d側はpadding=0
        return F.conv3d(
            x,
            self.weight,
            stride=1,
            padding=0
        )

# 例：データ読み込み
data_path = r"C:\Users\ri0151fv\Saito\Data\TrainData_NoBed.npz"
data = np.load(data_path)

print(data.files)

arr = data["Train"]
x_np = arr[:, :, :, 0]  # (128, 256, 256)

x = torch.tensor(
    x_np,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)

print("元画像:", x.shape)


# =====================================================
# 元画像を保存
# =====================================================

save_x_path = r"C:\Users\ri0151fv\Saito\original_input.npy"

np.save(
    save_x_path,
    x.detach().cpu().numpy()
)

print("元画像を保存:", save_x_path)


# =====================================================
# Analysisフィルタ
# =====================================================

analysis = Haar3DAnalysisOnly().to(device)
w = analysis(x)

print("Analysisフィルタ後:", w.shape)
print(analysis.names)

# Analysisフィルタ後のテンソルを保存
save_w_path = r"C:\Users\ri0151fv\Saito\analysis_output.npy"
np.save(save_w_path, w.detach().cpu().numpy())
print("Analysis出力を保存:", save_w_path)

# 元画像表示
slice_x = x.shape[2] // 2

plt.figure(figsize=(6, 6))
plt.imshow(x[0, 0, slice_x].detach().cpu().numpy(), cmap="gray")
plt.title("Original CT slice")
plt.axis("off")
plt.show()


# Analysisフィルタ後の8成分表示
slice_w = w.shape[2] // 2

plt.figure(figsize=(16, 8))

for i, name in enumerate(analysis.names):
    plt.subplot(2, 4, i + 1)
    plt.imshow(w[0, i, slice_w].detach().cpu().numpy(), cmap="gray")
    plt.title(name)
    plt.axis("off")

plt.show()