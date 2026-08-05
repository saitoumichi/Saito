"""DWT学習済みチェックポイントごとの8バンド可視化（読み取り専用）。"""

import math
import re
import sys
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# "dwt" は通常のCopy1、"inverse_consistency" は逆整合性版を評価する。
EXPERIMENT = "dwt"
PATIENT_ID = 0
# 見やすい固定テスト。3DランダムDVF評価にしたい場合は TEST_MODE を "random_3d" に変更する。
TEST_MODE = "constant_x_shift"
SHIFT_PIXELS = 20
RANDOM_SEED = 20260803
RANDOM_MAX_SHIFT = 40.0

SAITO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SAITO_DIR.parent))
import voxelmorph as vxm  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = SAITO_DIR / "Data" / "TrainData_NoBed.npz"
NB_FEATURES = [[32, 64, 64, 64, 64], [64, 64, 64, 64, 64, 32, 16, 16]]
BAND_NAMES = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]


class Haar3DAnalysis(nn.Module):
    def __init__(self):
        super().__init__()
        low = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
        high = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
        filters = []
        for z in (low, high):
            for y in (low, high):
                for x in (low, high):
                    filters.append(z[:, None, None] * y[None, :, None] * x[None, None, :])
        self.register_buffer("weight", torch.stack(filters).unsqueeze(1).float())

    def forward(self, image):
        return F.conv3d(F.pad(image, (0, 1, 0, 1, 0, 1)), self.weight)


def dwt3d(image, analysis):
    return analysis(image)[:, :, ::2, ::2, ::2]


def make_synthesis_filters():
    low = torch.tensor([1.0, 1.0], device=DEVICE) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], device=DEVICE) / math.sqrt(2.0)
    filters = []
    for z in (low, high):
        for y in (low, high):
            for x in (low, high):
                filters.append(z[:, None, None] * y[None, :, None] * x[None, None, :])
    return torch.flip(torch.stack(filters), dims=[1, 2, 3]).unsqueeze(1)


def inverse_dwt_with_bands(coefficients, synthesis_filters):
    """最終画像だけでなく、逆DWTにおける各バンドの寄与も返す。"""
    batch, channels, depth, height, width = coefficients.shape
    upsampled = torch.zeros(
        batch, channels, depth * 2, height * 2, width * 2,
        dtype=coefficients.dtype, device=coefficients.device,
    )
    upsampled[:, :, ::2, ::2, ::2] = coefficients
    contributions = []
    for index in range(channels):
        contribution = F.conv3d(upsampled[:, index:index + 1], synthesis_filters[index:index + 1], padding=1)
        contributions.append(contribution[:, :, :depth * 2, :height * 2, :width * 2])
    contributions = torch.cat(contributions, dim=1)
    return contributions.sum(dim=1, keepdim=True), contributions


def experiment_paths():
    if EXPERIMENT == "inverse_consistency":
        checkpoints = SAITO_DIR / "curriculum_inverse_consistency_checkpoints"
        fallback = SAITO_DIR / "model_analysis_pipeline_pretrain_inverse_consistency.pth"
        output = SAITO_DIR / "dwt_band_visualization_inverse_consistency"
    elif EXPERIMENT == "dwt":
        checkpoints = SAITO_DIR / "curriculum_checkpoints"
        fallback = SAITO_DIR / "model_analysis_pipeline_pretrain.pth"
        output = SAITO_DIR / "dwt_band_visualization"
    else:
        raise ValueError('EXPERIMENT must be "dwt" or "inverse_consistency"')
    return checkpoints, fallback, output


def discover_checkpoints(checkpoint_dir, fallback_path):
    paths = sorted(checkpoint_dir.glob("*.pth"))
    if paths:
        return paths
    if fallback_path.exists():
        print("2,000 epochごとのチェックポイントがないため、最新重みのみ可視化します。")
        return [fallback_path]
    raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_dir}")


def epoch_from_checkpoint(path, checkpoint):
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        return int(checkpoint["epoch"])
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model = vxm.networks.VxmDense_128_256_256((128, 256, 256), NB_FEATURES, int_steps=0).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, epoch_from_checkpoint(path, checkpoint)


def make_target(moving, fullres_transformer):
    if TEST_MODE == "constant_x_shift":
        flow = torch.zeros((1, 3, 128, 256, 256), device=DEVICE)
        flow[:, 2] = float(SHIFT_PIXELS)
        label = f"constant_x_{SHIFT_PIXELS:+d}px"
    elif TEST_MODE == "random_3d":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(RANDOM_SEED)
        flow = (torch.rand((1, 3, 8, 16, 16), generator=generator) * 2.0 - 1.0) * RANDOM_MAX_SHIFT
        from scipy.ndimage import gaussian_filter
        flow = torch.from_numpy(gaussian_filter(flow.numpy(), sigma=[0, 0, 2.0, 2.0, 2.0])).float()
        flow = F.interpolate(flow, size=(128, 256, 256), mode="trilinear", align_corners=False).to(DEVICE)
        label = f"random3d_seed_{RANDOM_SEED}"
    else:
        raise ValueError('TEST_MODE must be "constant_x_shift" or "random_3d"')
    return fullres_transformer(moving, flow), label


def plot_grid(volume, title, path, signed=True):
    """8バンドを2x4で保存。高周波の正負値はゼロ中心で表示する。"""
    middle = volume.shape[2] // 2
    images = volume[0, :, middle].detach().cpu().numpy()
    abs_max = max(float(np.abs(images).max()), 1e-8)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if signed:
            image = axis.imshow(images[index], cmap="coolwarm", norm=colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max))
        else:
            image = axis.imshow(images[index], cmap="gray")
        axis.set_title(BAND_NAMES[index])
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title, fontsize=16)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_reconstruction(moving, target, moved, path, epoch, label):
    middle = moving.shape[2] // 2
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, image, title in zip(axes[:3], [moving, target, moved], ["Moving", "Target", "Moved"]):
        axis.imshow(image[0, 0, middle].detach().cpu(), cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    error = (target - moved)[0, 0, middle].detach().cpu()
    error_image = axes[3].imshow(error, cmap="coolwarm", norm=colors.TwoSlopeNorm(vcenter=0))
    axes[3].set_title("Target − Moved")
    axes[3].axis("off")
    figure.colorbar(error_image, ax=axes[3], fraction=0.046, pad=0.04)
    figure.suptitle(f"Epoch {epoch}: {label}", fontsize=16)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    checkpoint_dir, fallback_path, output_dir = experiment_paths()
    output_dir.mkdir(exist_ok=True)
    x_train = np.transpose(np.load(DATA_PATH)["Train"], (3, 0, 1, 2))
    moving = torch.from_numpy(x_train[PATIENT_ID:PATIENT_ID + 1]).unsqueeze(1).float().to(DEVICE)

    analysis = Haar3DAnalysis().to(DEVICE)
    synthesis_filters = make_synthesis_filters()
    lowres_transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(DEVICE)
    fullres_transformer = vxm.layers.SpatialTransformer((128, 256, 256)).to(DEVICE)
    target, label = make_target(moving, fullres_transformer)

    for checkpoint_path in discover_checkpoints(checkpoint_dir, fallback_path):
        model, epoch = load_model(checkpoint_path)
        epoch_dir = output_dir / f"epoch_{epoch:06d}"
        epoch_dir.mkdir(exist_ok=True)
        print(f"Visualizing epoch {epoch}: {checkpoint_path.name}")
        with torch.no_grad():
            moving_bands = dwt3d(moving, analysis)
            target_bands = dwt3d(target, analysis)
            predicted_flow = model(moving_bands, target_bands)
            warped_bands = lowres_transformer(moving_bands, predicted_flow)
            moved, synthesis_contributions = inverse_dwt_with_bands(warped_bands, synthesis_filters)

        plot_grid(moving_bands, f"Epoch {epoch}: DWT bands of Moving", epoch_dir / "01_moving_dwt_bands.png")
        plot_grid(target_bands, f"Epoch {epoch}: DWT bands of Target", epoch_dir / "02_target_dwt_bands.png")
        plot_grid(warped_bands, f"Epoch {epoch}: warped DWT bands", epoch_dir / "03_warped_dwt_bands.png")
        plot_grid(synthesis_contributions, f"Epoch {epoch}: inverse-DWT band contributions", epoch_dir / "04_synthesis_contributions.png")
        plot_reconstruction(moving, target, moved, epoch_dir / "05_reconstruction.png", epoch, label)

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
