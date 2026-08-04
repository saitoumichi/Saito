"""保存済みカリキュラム学習チェックポイントの後評価。

学習を止めたり、重みを書き換えたりせずに実行できる。
評価対象は、正負x方向の一定シフトと、固定seedの滑らかなランダム3D DVF。
"""

import csv
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# このスクリプトを Saito フォルダに置いて実行する前提。
SAITO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SAITO_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import voxelmorph as vxm  # noqa: E402


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = SAITO_DIR / "Data" / "TrainData_NoBed.npz"
CHECKPOINT_DIR = SAITO_DIR / "curriculum_checkpoints"
FALLBACK_MODEL_PATH = SAITO_DIR / "model_analysis_pipeline_pretrain.pth"
OUTPUT_DIR = SAITO_DIR / "posthoc_curriculum_evaluation"

PATIENT_ID = 0
SHIFTS = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
RANDOM_SEED = 20260803
RANDOM_MAX_SHIFT = 40.0

NB_FEATURES = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16],
]


class Haar3DAnalysis(nn.Module):
    """現在の学習ノートブックと同じ、1段の3D直交Haar DWT。"""

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

    def forward(self, x):
        return F.conv3d(F.pad(x, (0, 1, 0, 1, 0, 1)), self.weight)


def dwt3d(x, analysis):
    return analysis(x)[:, :, ::2, ::2, ::2]


def idwt3d(coefficients, synthesis_filters):
    """現在のノートブックの up_sampling_3d + synthesis_filter_3d と同じ処理。"""
    batch, channels, depth, height, width = coefficients.shape
    upsampled = torch.zeros(
        batch, channels, depth * 2, height * 2, width * 2,
        dtype=coefficients.dtype, device=coefficients.device,
    )
    upsampled[:, :, ::2, ::2, ::2] = coefficients
    bands = []
    for band_index in range(channels):
        filtered = F.conv3d(upsampled[:, band_index:band_index + 1], synthesis_filters[band_index:band_index + 1], padding=1)
        bands.append(filtered[:, :, :depth * 2, :height * 2, :width * 2])
    return torch.cat(bands, dim=1).sum(dim=1, keepdim=True)


def make_synthesis_filters(device):
    low = torch.tensor([1.0, 1.0], device=device) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], device=device) / math.sqrt(2.0)
    filters = []
    for z in (low, high):
        for y in (low, high):
            for x in (low, high):
                filters.append(z[:, None, None] * y[None, :, None] * x[None, None, :])
    return torch.flip(torch.stack(filters), dims=[1, 2, 3]).unsqueeze(1)


def rmse(reference, prediction):
    return torch.sqrt(torch.mean((reference - prediction) ** 2)).item()


def discover_checkpoints():
    checkpoints = sorted(CHECKPOINT_DIR.glob("pretrain_epoch_*.pth"))
    if checkpoints:
        return checkpoints
    if FALLBACK_MODEL_PATH.exists():
        print("curriculum_checkpoints がないため、最新重みだけを評価します。")
        return [FALLBACK_MODEL_PATH]
    raise FileNotFoundError(f"評価対象がありません: {CHECKPOINT_DIR} または {FALLBACK_MODEL_PATH}")


def checkpoint_epoch(path, checkpoint):
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        return int(checkpoint["epoch"])
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model = vxm.networks.VxmDense_128_256_256((128, 256, 256), NB_FEATURES, int_steps=0).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint_epoch(path, checkpoint)


def register(model, moving, target, analysis, synthesis_filters, transformer):
    moving_w = dwt3d(moving, analysis)
    target_w = dwt3d(target, analysis)
    predicted_flow = model(moving_w, target_w)
    warped_coefficients = transformer(moving_w, predicted_flow)
    moved = idwt3d(warped_coefficients, synthesis_filters)
    return moved, predicted_flow


def fixed_random_flow():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_SEED)
    flow = (torch.rand((1, 3, 8, 16, 16), generator=generator) * 2 - 1) * RANDOM_MAX_SHIFT
    # 学習時と同じGaussian smoothing。CPUで行った後にGPUへ送る。
    from scipy.ndimage import gaussian_filter
    flow = torch.from_numpy(gaussian_filter(flow.numpy(), sigma=[0, 0, 2.0, 2.0, 2.0])).float()
    return F.interpolate(flow, size=(128, 256, 256), mode="trilinear", align_corners=False).to(DEVICE)


def evaluate_checkpoint(model, epoch, moving, analysis, synthesis_filters, transformer, transformer256, random_flow):
    rows = []
    with torch.no_grad():
        for shift in SHIFTS:
            true_flow_full = torch.zeros((1, 3, 128, 256, 256), device=DEVICE)
            true_flow_full[:, 2] = float(shift)
            target = transformer256(moving, true_flow_full)
            moved, predicted_flow = register(model, moving, target, analysis, synthesis_filters, transformer)
            true_flow_low = torch.zeros_like(predicted_flow)
            true_flow_low[:, 2] = shift / 2.0
            rows.append({
                "epoch": epoch,
                "test": "constant_x_shift",
                "shift_pixels_fullres_x": shift,
                "rmse_before": rmse(target, moving),
                "rmse_after": rmse(target, moved),
                "dvf_rmse": rmse(true_flow_low, predicted_flow),
                "predicted_x_mean_lowres": predicted_flow[:, 2].mean().item(),
            })

        random_target = transformer256(moving, random_flow)
        random_moved, random_predicted_flow = register(
            model, moving, random_target, analysis, synthesis_filters, transformer
        )
        random_true_flow_low = F.interpolate(
            random_flow, size=random_predicted_flow.shape[2:], mode="trilinear", align_corners=False
        )
        rows.append({
            "epoch": epoch,
            "test": "fixed_random_smooth_dvf",
            "shift_pixels_fullres_x": "",
            "rmse_before": rmse(random_target, moving),
            "rmse_after": rmse(random_target, random_moved),
            "dvf_rmse": rmse(random_true_flow_low, random_predicted_flow),
            "predicted_x_mean_lowres": random_predicted_flow[:, 2].mean().item(),
        })
    return rows


def save_plots(rows):
    constant_rows = [row for row in rows if row["test"] == "constant_x_shift"]
    random_rows = [row for row in rows if row["test"] == "fixed_random_smooth_dvf"]
    epochs = sorted({row["epoch"] for row in constant_rows})

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for epoch in epochs:
        subset = sorted((row for row in constant_rows if row["epoch"] == epoch), key=lambda row: row["shift_pixels_fullres_x"])
        axes[0].plot([row["shift_pixels_fullres_x"] for row in subset], [row["rmse_after"] for row in subset], marker="o", label=f"epoch {epoch}")
        axes[1].plot([row["shift_pixels_fullres_x"] for row in subset], [row["dvf_rmse"] for row in subset], marker="o", label=f"epoch {epoch}")
    axes[0].set(title="Image RMSE after registration", xlabel="True x shift (full-resolution pixels)", ylabel="RMSE")
    axes[1].set(title="DVF RMSE", xlabel="True x shift (full-resolution pixels)", ylabel="RMSE")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "constant_shift_curves.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([row["epoch"] for row in random_rows], [row["rmse_after"] for row in random_rows], marker="o")
    axes[1].plot([row["epoch"] for row in random_rows], [row["dvf_rmse"] for row in random_rows], marker="o")
    axes[0].set(title="Fixed random DVF: image RMSE", xlabel="Checkpoint epoch", ylabel="RMSE")
    axes[1].set(title="Fixed random DVF: DVF RMSE", xlabel="Checkpoint epoch", ylabel="RMSE")
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "random_dvf_curves.png", dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x_train = np.load(DATA_PATH)["Train"]
    x_train = np.transpose(x_train, (3, 0, 1, 2))
    moving = torch.from_numpy(x_train[PATIENT_ID:PATIENT_ID + 1]).unsqueeze(1).float().to(DEVICE)

    analysis = Haar3DAnalysis().to(DEVICE)
    synthesis_filters = make_synthesis_filters(DEVICE)
    transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(DEVICE)
    transformer256 = vxm.layers.SpatialTransformer((128, 256, 256)).to(DEVICE)
    random_flow = fixed_random_flow()

    rows = []
    for path in discover_checkpoints():
        model, epoch = load_model(path)
        print(f"Evaluating epoch {epoch}: {path.name}")
        rows.extend(evaluate_checkpoint(model, epoch, moving, analysis, synthesis_filters, transformer, transformer256, random_flow))

    csv_path = OUTPUT_DIR / "posthoc_curriculum_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_plots(rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {OUTPUT_DIR / 'constant_shift_curves.png'}")
    print(f"Saved: {OUTPUT_DIR / 'random_dvf_curves.png'}")


if __name__ == "__main__":
    main()
