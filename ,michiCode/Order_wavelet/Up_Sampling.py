import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Downsampling.py で保存した w_down を読み込む
w_down_path = r"C:\Users\ri0151fv\Saito\analysis_downsampled.npy"
w_down_np = np.load(w_down_path)

w_down = torch.tensor(
    w_down_np,
    dtype=torch.float32
).to(device)

print("Downsampling後:", w_down.shape)


# =====================================================
# アップサンプリング
# 各軸方向で1つおきに元データを配置し、
# その間を0で埋める
# =====================================================

B, C, D, H, W = w_down.shape

w_up = torch.zeros(
    B,
    C,
    D * 2,
    H * 2,
    W * 2,
    dtype=w_down.dtype,
    device=device
)

# 偶数位置に元の値を配置
w_up[:, :, ::2, ::2, ::2] = w_down

print("Upsampling後:", w_up.shape)


# =====================================================
# 保存
# =====================================================

save_path = r"C:\Users\ri0151fv\Saito\upsampled_output.npy"

np.save(
    save_path,
    w_up.detach().cpu().numpy()
)

print("保存先:", save_path)


# =====================================================
# 表示
# =====================================================

names = [
    "LLL",
    "LLH",
    "LHL",
    "LHH",
    "HLL",
    "HLH",
    "HHL",
    "HHH"
]

slice_up = w_up.shape[2] // 2

if slice_up % 2 != 0:
    slice_up -= 1
    
plt.figure(figsize=(16, 8))

for i, name in enumerate(names):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        w_up[0, i, slice_up]
        .detach()
        .cpu()
        .numpy(),
        cmap="gray"
    )

    plt.title(name + " upsampled")
    plt.axis("off")


save_fig_path = r"C:\Users\ri0151fv\Saito\upsampled_8bands.png"

plt.savefig(
    save_fig_path,
    dpi=300,
    bbox_inches="tight"
)

print("画像保存先:", save_fig_path)

plt.close()

# 元のダウンサンプリング結果
print(w_down.shape)

# アップサンプリング結果
print(w_up.shape)

# 元データが偶数位置に正しく保存されているか
error = torch.abs(
    w_up[:, :, ::2, ::2, ::2] - w_down
)

print("平均誤差:", error.mean().item())
print("最大誤差:", error.max().item())