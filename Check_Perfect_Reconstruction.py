import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


# =====================================================
# デバイス設定
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# =====================================================
# 共通設定
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


# =====================================================
# 1. Haar 3D Analysis Filter
# =====================================================

class Haar3DAnalysis(nn.Module):

    def __init__(self):
        super().__init__()

        # 1次元Haarフィルタ
        hL = torch.tensor(
            [1.0, 1.0],
            dtype=torch.float32
        ) / math.sqrt(2)

        hH = torch.tensor(
            [1.0, -1.0],
            dtype=torch.float32
        ) / math.sqrt(2)


        filters = []
        filter_names = []


        # =================================================
        # 3方向に L / H を組み合わせ、
        # 8個の3Dフィルタを作成
        # =================================================

        for z_name, z_filter in zip(
            ["L", "H"],
            [hL, hH]
        ):

            for y_name, y_filter in zip(
                ["L", "H"],
                [hL, hH]
            ):

                for x_name, x_filter in zip(
                    ["L", "H"],
                    [hL, hH]
                ):

                    kernel = (
                        z_filter[:, None, None]
                        * y_filter[None, :, None]
                        * x_filter[None, None, :]
                    )

                    filters.append(kernel)

                    filter_names.append(
                        z_name
                        + y_name
                        + x_name
                    )


        # shape:
        # (8, 1, 2, 2, 2)

        weight = (
            torch.stack(
                filters,
                dim=0
            )
            .unsqueeze(1)
        )


        self.register_buffer(
            "weight",
            weight
        )

        self.names = filter_names


    def forward(self, x):

        # =================================================
        # 各軸の末尾側に1ボクセルpadding
        #
        # Input:
        # 128 × 256 × 256
        #
        # Padding:
        # 129 × 257 × 257
        #
        # Conv kernel=2:
        # 128 × 256 × 256
        # =================================================

        x_pad = F.pad(
            x,
            (
                0, 1,   # W
                0, 1,   # H
                0, 1    # D
            )
        )


        w = F.conv3d(
            x_pad,
            self.weight,
            stride=1,
            padding=0
        )


        return w


# =====================================================
# 2. Downsampling
# =====================================================

def downsample_3d(w):

    # 各軸方向で2個に1個だけ残す

    w_down = w[
        :,
        :,
        ::2,
        ::2,
        ::2
    ]

    return w_down


# =====================================================
# 3. Upsampling
# =====================================================

def upsample_3d(w_down):

    B, C, D, H, W = w_down.shape


    # 元の2倍サイズのゼロテンソルを作成

    w_up = torch.zeros(
        B,
        C,
        D * 2,
        H * 2,
        W * 2,
        dtype=w_down.dtype,
        device=w_down.device
    )


    # =================================================
    # 偶数位置に値を配置
    #
    # 1Dの例:
    #
    # [a, b, c]
    #
    # ↓
    #
    # [a, 0, b, 0, c, 0]
    # =================================================

    w_up[
        :,
        :,
        ::2,
        ::2,
        ::2
    ] = w_down


    return w_up


# =====================================================
# 4. 3D Synthesis Filter作成関数
# =====================================================

def make_3d_filter(fz, fy, fx):

    kernel = (
        fz[:, None, None]
        * fy[None, :, None]
        * fx[None, None, :]
    )

    return kernel


# =====================================================
# 5. Haar 3D Synthesis Filter
# =====================================================

class Haar3DSynthesis(nn.Module):

    def __init__(self):
        super().__init__()


        sqrt2 = math.sqrt(2.0)


        low = torch.tensor(
            [1.0, 1.0],
            dtype=torch.float32
        ) / sqrt2


        high = torch.tensor(
            [1.0, -1.0],
            dtype=torch.float32
        ) / sqrt2


        # =================================================
        # 8種類の3D Synthesis Filter
        # =================================================

        filters = torch.stack(
            [

                make_3d_filter(
                    low,
                    low,
                    low
                ),  # LLL


                make_3d_filter(
                    low,
                    low,
                    high
                ),  # LLH


                make_3d_filter(
                    low,
                    high,
                    low
                ),  # LHL


                make_3d_filter(
                    low,
                    high,
                    high
                ),  # LHH


                make_3d_filter(
                    high,
                    low,
                    low
                ),  # HLL


                make_3d_filter(
                    high,
                    low,
                    high
                ),  # HLH


                make_3d_filter(
                    high,
                    high,
                    low
                ),  # HHL


                make_3d_filter(
                    high,
                    high,
                    high
                )   # HHH

            ],
            dim=0
        )


        # =================================================
        # PyTorchのconv3dは相関演算なので
        # 数学的な畳み込み用に反転
        # =================================================

        filters = torch.flip(
            filters,
            dims=[1, 2, 3]
        )


        # shape:
        # (8, 1, 2, 2, 2)

        filters = filters.unsqueeze(1)


        self.register_buffer(
            "filters",
            filters
        )


    def forward(self, w_up):

        B, C, D, H, W = w_up.shape


        filtered_bands = []


        # =================================================
        # 8成分それぞれに対応する
        # Synthesis Filterを適用
        # =================================================

        for i, name in enumerate(names):


            # 1成分だけ取り出す

            band = w_up[
                :,
                i:i + 1,
                :,
                :,
                :
            ]


            # 対応するフィルタ

            kernel = self.filters[
                i:i + 1
            ]


            # Synthesis filtering

            filtered = F.conv3d(
                band,
                kernel,
                stride=1,
                padding=1
            )


            # =================================================
            # paddingによりサイズが1大きくなるため、
            # Upsampling後のサイズへ合わせる
            # =================================================

            filtered = filtered[
                :,
                :,
                :D,
                :H,
                :W
            ]


            filtered_bands.append(
                filtered
            )


            print(
                name,
                "Synthesis後:",
                filtered.shape
            )


        # =================================================
        # 8成分をチャンネル方向へ結合
        # =================================================

        filtered_bands = torch.cat(
            filtered_bands,
            dim=1
        )


        # =================================================
        # 8成分を加算
        # =================================================

        reconstructed = torch.sum(
            filtered_bands,
            dim=1,
            keepdim=True
        )


        return (
            filtered_bands,
            reconstructed
        )


# =====================================================
# CTデータ読み込み
# =====================================================

data_path = (
    r"C:\Users\ri0151fv\Saito\Data\TrainData_NoBed.npz"
)


data = np.load(
    data_path
)


print(
    "npz keys:",
    data.files
)


arr = data["Train"]


# 1症例を使用

x_np = arr[
    :,
    :,
    :,
    0
]


x = torch.tensor(
    x_np,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)


print("\n===================================")
print("入力")
print("===================================")

print(
    "元画像:",
    x.shape
)


# =====================================================
# Analysis Filter
# =====================================================

print("\n===================================")
print("1. Analysis Filter")
print("===================================")


analysis = Haar3DAnalysis().to(device)


w = analysis(x)


print(
    "Analysis後:",
    w.shape
)


print(
    "周波数成分:",
    analysis.names
)


# =====================================================
# Downsampling
# =====================================================

print("\n===================================")
print("2. Downsampling")
print("===================================")


w_down = downsample_3d(
    w
)


print(
    "Downsampling後:",
    w_down.shape
)


# =====================================================
# Upsampling
# =====================================================

print("\n===================================")
print("3. Upsampling")
print("===================================")


w_up = upsample_3d(
    w_down
)


print(
    "Upsampling後:",
    w_up.shape
)


# =====================================================
# Upsampling確認
# =====================================================

up_error = torch.abs(
    w_up[
        :,
        :,
        ::2,
        ::2,
        ::2
    ]
    - w_down
)


print(
    "Upsampling配置確認 平均誤差:",
    up_error.mean().item()
)


print(
    "Upsampling配置確認 最大誤差:",
    up_error.max().item()
)


# =====================================================
# Synthesis Filter
# =====================================================

print("\n===================================")
print("4. Synthesis Filter")
print("===================================")


synthesis = Haar3DSynthesis().to(device)


filtered_bands, reconstructed = synthesis(
    w_up
)


print(
    "\nSynthesis後8成分:",
    filtered_bands.shape
)


print(
    "再構成画像:",
    reconstructed.shape
)


# =====================================================
# 再構成誤差確認
# =====================================================

print("\n===================================")
print("5. Reconstruction Error")
print("===================================")


if x.shape != reconstructed.shape:

    print(
        "サイズが一致していません"
    )

    print(
        "Original:",
        x.shape
    )

    print(
        "Reconstructed:",
        reconstructed.shape
    )


else:

    print(
        "サイズ一致"
    )


    diff = (
        x
        - reconstructed
    )


    absolute_error = torch.abs(
        diff
    )


    mae = torch.mean(
        absolute_error
    )


    max_error = torch.max(
        absolute_error
    )


    mse = torch.mean(
        diff ** 2
    )


    relative_error = (
        torch.norm(diff)
        /
        torch.norm(x)
    )


    print(
        "\n===== 再構成誤差 ====="
    )


    print(
        "MAE:",
        mae.item()
    )


    print(
        "最大絶対誤差:",
        max_error.item()
    )


    print(
        "MSE:",
        mse.item()
    )


    print(
        "相対誤差:",
        relative_error.item()
    )


    if max_error.item() > 0:

        error_order = int(
            np.floor(
                np.log10(
                    max_error.item()
                )
            )
        )

        error_scale = 10.0 ** (-error_order)

    else:

        error_order = 0
        error_scale = 1.0


    print(
        "誤差表示スケール:",
        f"x 1e{-error_order}"
        if max_error.item() > 0
        else "no scaling"
    )


# =====================================================
# 結果保存
# =====================================================

print("\n===================================")
print("6. Save Results")
print("===================================")


np.save(
    r"C:\Users\ri0151fv\Saito\wavelet_reconstructed.npy",
    reconstructed
    .detach()
    .cpu()
    .numpy()
)


print(
    "再構成画像を保存しました"
)


# =====================================================
# 元画像・再構成画像・誤差画像を表示
# =====================================================

if x.shape == reconstructed.shape:


    slice_index = (
        x.shape[2] // 2
    )


    # =================================================
    # 元画像
    # =================================================

    plt.figure(
        figsize=(8, 8)
    )


    plt.imshow(
        x[
            0,
            0,
            slice_index
        ]
        .detach()
        .cpu()
        .numpy(),
        cmap="gray"
    )


    plt.title(
        "Original Image"
    )


    plt.axis(
        "off"
    )


    plt.savefig(
        r"C:\Users\ri0151fv\Saito\wavelet_original.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =================================================
    # 再構成画像
    # =================================================

    plt.figure(
        figsize=(8, 8)
    )


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


    plt.title(
        "Reconstructed Image"
    )


    plt.axis(
        "off"
    )


    plt.savefig(
        r"C:\Users\ri0151fv\Saito\wavelet_reconstructed.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =================================================
    # 誤差画像（最大誤差基準）
    # =================================================

    plt.figure(
        figsize=(8, 8)
    )


    plt.imshow(
        absolute_error[
            0,
            0,
            slice_index
        ]
        .detach()
        .cpu()
        .numpy(),
        cmap="inferno",
        vmin=0.0,
        vmax=max_error.item()
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04
    )


    plt.title(
        "Absolute Reconstruction Error"
    )


    plt.axis(
        "off"
    )


    plt.savefig(
        r"C:\Users\ri0151fv\Saito\wavelet_error.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =================================================
    # 誤差画像（拡大表示）
    # =================================================

    plt.figure(
        figsize=(8, 8)
    )


    plt.imshow(
        (
            absolute_error[
                0,
                0,
                slice_index
            ]
            * error_scale
        )
        .detach()
        .cpu()
        .numpy(),
        cmap="inferno"
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04
    )


    plt.title(
        f"Scaled Error (x 1e{-error_order})"
        if max_error.item() > 0
        else "Scaled Error"
    )


    plt.axis(
        "off"
    )


    plt.savefig(
        r"C:\Users\ri0151fv\Saito\wavelet_error_scaled.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()

# =====================================================
# 7. Perfect Reconstruction Conditions Check
# =====================================================

print("\n===================================")
print("7. Perfect Reconstruction Conditions")
print("===================================")


# =====================================================
# 実装上の1D Analysis / Synthesisフィルタ
# =====================================================
#
# Analysis側のconv3dは相関演算なので、
# 数学的な畳み込みフィルタとして見ると
# High-pass側は反転した形になる
#
# Analysis:
# H0 = [ 1,  1] / sqrt(2)
# H1 = [-1,  1] / sqrt(2)
#
# Synthesis:
# F0 = [ 1,  1] / sqrt(2)
# F1 = [ 1, -1] / sqrt(2)
# =====================================================

sqrt2 = np.sqrt(2.0)

h0 = np.array(
    [1.0, 1.0]
) / sqrt2

h1 = np.array(
    [-1.0, 1.0]
) / sqrt2

f0 = np.array(
    [1.0, 1.0]
) / sqrt2

f1 = np.array(
    [1.0, -1.0]
) / sqrt2


# =====================================================
# 周波数軸
# =====================================================

N = 2048

omega = np.linspace(
    -np.pi,
    np.pi,
    N,
    endpoint=False
)


# =====================================================
# DTFT計算関数
# =====================================================

def frequency_response(h, omega):

    n = np.arange(
        len(h)
    )

    response = np.sum(
        h[None, :]
        * np.exp(
            -1j
            * omega[:, None]
            * n[None, :]
        ),
        axis=1
    )

    return response


# =====================================================
# フィルタの周波数応答
# =====================================================

H0 = frequency_response(
    h0,
    omega
)

H1 = frequency_response(
    h1,
    omega
)

F0 = frequency_response(
    f0,
    omega
)

F1 = frequency_response(
    f1,
    omega
)


# H(-z) に対応
# 周波数領域では ω + π

H0_minus = frequency_response(
    h0,
    omega + np.pi
)

H1_minus = frequency_response(
    h1,
    omega + np.pi
)


# =====================================================
# 条件1
# Alias Cancellation
# =====================================================

alias_term = (
    H0_minus * F0
    +
    H1_minus * F1
)

alias_max = np.max(
    np.abs(alias_term)
)


print("\n-----------------------------------")
print("条件1: Alias Cancellation")
print("-----------------------------------")

print(
    "最大Alias成分:",
    alias_max
)


# =====================================================
# 信号伝達関数
# =====================================================

T = (
    H0 * F0
    +
    H1 * F1
)


# =====================================================
# 条件2
# Amplitude Distortion
# =====================================================

magnitude = np.abs(T)

magnitude_min = np.min(
    magnitude
)

magnitude_max = np.max(
    magnitude
)

magnitude_variation = (
    magnitude_max
    -
    magnitude_min
)


print("\n-----------------------------------")
print("条件2: Amplitude Distortion")
print("-----------------------------------")

print(
    "Magnitude min:",
    magnitude_min
)

print(
    "Magnitude max:",
    magnitude_max
)

print(
    "Magnitude variation:",
    magnitude_variation
)


# =====================================================
# 条件3
# Phase Distortion
# =====================================================

phase = np.unwrap(
    np.angle(T)
)


# 位相を直線近似

phase_coef = np.polyfit(
    omega,
    phase,
    1
)


phase_fit = np.polyval(
    phase_coef,
    omega
)


phase_error = (
    phase
    -
    phase_fit
)


max_phase_error = np.max(
    np.abs(phase_error)
)


print("\n-----------------------------------")
print("条件3: Phase Distortion")
print("-----------------------------------")

print(
    "Phase slope:",
    phase_coef[0]
)

print(
    "最大直線位相誤差:",
    max_phase_error
)


# =====================================================
# 判定
# =====================================================

tolerance = 1e-10


print("\n===================================")
print("PR Condition Results")
print("===================================")


if alias_max < tolerance:

    print(
        "条件1 Alias Cancellation: OK"
    )

else:

    print(
        "条件1 Alias Cancellation: NG"
    )


if magnitude_variation < tolerance:

    print(
        "条件2 Amplitude Distortion: OK"
    )

else:

    print(
        "条件2 Amplitude Distortion: NG"
    )


if max_phase_error < tolerance:

    print(
        "条件3 Phase Distortion: OK"
    )

else:

    print(
        "条件3 Phase Distortion: NG"
    )


# =====================================================
# Alias成分のグラフ
# =====================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    omega,
    np.abs(alias_term)
)

plt.xlabel(
    "Angular Frequency ω"
)

plt.ylabel(
    "|Alias Term|"
)

plt.title(
    "Alias Cancellation"
)

plt.grid()

plt.savefig(
    r"C:\Users\ri0151fv\Saito\pr_alias.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 振幅応答
# =====================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    omega,
    magnitude
)

plt.xlabel(
    "Angular Frequency ω"
)

plt.ylabel(
    "|T(e^jω)|"
)

plt.title(
    "Distortion Transfer Function Magnitude"
)

plt.grid()

plt.savefig(
    r"C:\Users\ri0151fv\Saito\pr_amplitude.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 位相応答
# =====================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    omega,
    phase,
    label="Actual Phase"
)

plt.plot(
    omega,
    phase_fit,
    linestyle="--",
    label="Linear Fit"
)

plt.xlabel(
    "Angular Frequency ω"
)

plt.ylabel(
    "Phase [rad]"
)

plt.title(
    "Phase Response"
)

plt.legend()

plt.grid()

plt.savefig(
    r"C:\Users\ri0151fv\Saito\pr_phase.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nPR条件確認用グラフを保存しました"
)

print("\n処理完了")
