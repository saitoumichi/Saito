import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =====================================================
# デバイス設定
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# =====================================================
# Upsampling後のデータを読み込む
# =====================================================

input_path = r"C:\Users\ri0151fv\Saito\upsampled_output.npy"

w_up_np = np.load(input_path)

w_up = torch.tensor(
    w_up_np,
    dtype=torch.float32
).to(device)

print("Upsampling後:", w_up.shape)


# =====================================================
# Haar Synthesis Filter
# =====================================================

sqrt2 = np.sqrt(2.0)

low = torch.tensor(
    [1.0, 1.0],
    dtype=torch.float32,
    device=device
) / sqrt2

high = torch.tensor(
    [1.0, -1.0],
    dtype=torch.float32,
    device=device
) / sqrt2


# =====================================================
# 3Dフィルタ作成関数
# =====================================================

def make_3d_filter(fz, fy, fx):
    """
    1次元フィルタ3本から
    2×2×2 の3次元フィルタを作成
    """

    kernel = (
        fz[:, None, None]
        * fy[None, :, None]
        * fx[None, None, :]
    )

    return kernel


# =====================================================
# 8種類の3D Synthesis Filter
# =====================================================

filters = torch.stack([
    make_3d_filter(low,  low,  low),   # LLL
    make_3d_filter(low,  low,  high),  # LLH
    make_3d_filter(low,  high, low),   # LHL
    make_3d_filter(low,  high, high),  # LHH
    make_3d_filter(high, low,  low),   # HLL
    make_3d_filter(high, low,  high),  # HLH
    make_3d_filter(high, high, low),   # HHL
    make_3d_filter(high, high, high),  # HHH
], dim=0)


print("Synthesis filters:", filters.shape)


# =====================================================
# PyTorchのconv3dは相関演算なので
# 数学的な畳み込みとして使うためフィルタを反転
# =====================================================

filters = torch.flip(
    filters,
    dims=[1, 2, 3]
)


# shape:
# (8, 1, 2, 2, 2)

filters = filters.unsqueeze(1)

print("Conv用フィルタ:", filters.shape)


# =====================================================
# Synthesis Filter処理
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


B, C, D, H, W = w_up.shape

filtered_bands = []


for i, name in enumerate(names):

    # 1成分だけ取り出す
    band = w_up[:, i:i+1, :, :, :]

    # 対応するSynthesis Filter
    kernel = filters[i:i+1]

    # Synthesis filtering
    filtered = F.conv3d(
        band,
        kernel,
        stride=1,
        padding=1
    )

    # paddingによってサイズが1大きくなるので
    # Upsampling後のサイズに合わせる
    filtered = filtered[
        :,
        :,
        :D,
        :H,
        :W
    ]

    filtered_bands.append(filtered)

    print(
        name,
        "Synthesis後:",
        filtered.shape
    )


# =====================================================
# 8成分をチャンネル方向に結合
# =====================================================

filtered_bands = torch.cat(
    filtered_bands,
    dim=1
)

print(
    "Synthesis Filter後の8成分:",
    filtered_bands.shape
)


# =====================================================
# 8成分を加算
# =====================================================

reconstructed = torch.sum(
    filtered_bands,
    dim=1,
    keepdim=True
)

print(
    "再構成画像:",
    reconstructed.shape
)


# =====================================================
# 保存
# =====================================================

bands_save_path = (
    r"C:\Users\ri0151fv\Saito\synthesis_8bands.npy"
)

np.save(
    bands_save_path,
    filtered_bands
    .detach()
    .cpu()
    .numpy()
)

print(
    "8成分保存先:",
    bands_save_path
)


reconstructed_save_path = (
    r"C:\Users\ri0151fv\Saito\reconstructed_output.npy"
)

np.save(
    reconstructed_save_path,
    reconstructed
    .detach()
    .cpu()
    .numpy()
)

print(
    "再構成画像保存先:",
    reconstructed_save_path
)


# =====================================================
# 8成分を表示
# =====================================================

slice_index = filtered_bands.shape[2] // 2

plt.figure(figsize=(16, 8))

for i, name in enumerate(names):

    plt.subplot(2, 4, i + 1)

    img = (
        filtered_bands[
            0,
            i,
            slice_index
        ]
        .detach()
        .cpu()
        .numpy()
    )

    plt.imshow(
        img,
        cmap="gray"
    )

    plt.title(
        name + " synthesis"
    )

    plt.axis("off")


bands_fig_path = (
    r"C:\Users\ri0151fv\Saito\synthesis_8bands.png"
)

plt.savefig(
    bands_fig_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "8成分画像保存先:",
    bands_fig_path
)

plt.close()


# =====================================================
# 再構成画像を表示
# =====================================================

slice_index = reconstructed.shape[2] // 2

plt.figure(figsize=(8, 8))

plt.imshow(
    reconstructed[
        0,
        0,
        slice_index
    ]
    .detach()
    .cpu()
    .numpy(),
    cmap="gray"
)

plt.title("Reconstructed Image")

plt.axis("off")


reconstructed_fig_path = (
    r"C:\Users\ri0151fv\Saito\reconstructed_image.png"
)

plt.savefig(
    reconstructed_fig_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    "再構成画像保存先:",
    reconstructed_fig_path
)

plt.close()