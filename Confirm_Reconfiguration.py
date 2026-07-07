import numpy as np
import torch
import matplotlib.pyplot as plt


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# =====================================================
# 元画像を読み込む
# =====================================================

original_path = r"C:\Users\ri0151fv\Saito\original_input.npy"

original_np = np.load(original_path)

original = torch.tensor(
    original_np,
    dtype=torch.float32
).to(device)

print("元画像:", original.shape)


# =====================================================
# 再構成画像を読み込む
# =====================================================

reconstructed_path = (
    r"C:\Users\ri0151fv\Saito\reconstructed_output.npy"
)

reconstructed_np = np.load(reconstructed_path)

reconstructed = torch.tensor(
    reconstructed_np,
    dtype=torch.float32
).to(device)

print("再構成画像:", reconstructed.shape)


# =====================================================
# サイズ確認
# =====================================================

if original.shape != reconstructed.shape:

    print("\nサイズが一致していません")
    print("元画像:", original.shape)
    print("再構成:", reconstructed.shape)

else:

    print("\nサイズ一致")


    # =================================================
    # 誤差計算
    # =================================================

    diff = original - reconstructed

    error = torch.abs(diff)

    mae = error.mean()

    max_error = error.max()

    mse = torch.mean(diff ** 2)

    relative_error = (
        torch.norm(diff)
        / torch.norm(original)
    )


    print("\n===== 再構成誤差 =====")

    print(
        "平均絶対誤差:",
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


    # =================================================
    # 中央断面
    # =================================================

    slice_index = original.shape[2] // 2


    # =================================================
    # 元画像
    # =================================================

    plt.figure(figsize=(8, 8))

    plt.imshow(
        original[
            0,
            0,
            slice_index
        ].detach().cpu().numpy(),
        cmap="gray"
    )

    plt.title("Original Image")
    plt.axis("off")

    original_fig_path = (
        r"C:\Users\ri0151fv\Saito\check_original.png"
    )

    plt.savefig(
        original_fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =================================================
    # 再構成画像
    # =================================================

    plt.figure(figsize=(8, 8))

    plt.imshow(
        reconstructed[
            0,
            0,
            slice_index
        ].detach().cpu().numpy(),
        cmap="gray"
    )

    plt.title("Reconstructed Image")
    plt.axis("off")

    reconstructed_fig_path = (
        r"C:\Users\ri0151fv\Saito\check_reconstructed.png"
    )

    plt.savefig(
        reconstructed_fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =================================================
    # 誤差画像
    # 自動スケーリング
    # =================================================

    plt.figure(figsize=(8, 8))

    plt.imshow(
        error[
            0,
            0,
            slice_index
        ].detach().cpu().numpy(),
        cmap="gray"
    )

    plt.title("Absolute Error (Auto Scale)")
    plt.axis("off")

    error_fig_path = (
        r"C:\Users\ri0151fv\Saito\check_error_auto.png"
    )

    plt.savefig(
        error_fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =================================================
    # 誤差画像
    # 表示範囲を 0～1 に固定
    # =================================================

    plt.figure(figsize=(8, 8))

    plt.imshow(
        error[
            0,
            0,
            slice_index
        ].detach().cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title("Absolute Error (Range: 0-1)")
    plt.axis("off")

    error_fixed_fig_path = (
        r"C:\Users\ri0151fv\Saito\check_error_fixed.png"
    )

    plt.savefig(
        error_fixed_fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print("\n比較画像を保存しました")