"""256→128 Wavelet DVF pipeline verification.

This script never changes a training notebook or checkpoint.  It writes only
CSV/JSON/PNG results under ``dvf_wavelet_verification/``.

Run from the Saito directory on the Windows training machine:
    python verify_256_128_wavelet_dvf.py --steps 100

Use ``--steps 1000`` only after the 100-step comparison finishes.  Experiment
B starts the two scale conditions from identical weights and uses identical
synthetic samples at each step.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F


SAITO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SAITO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import voxelmorph as vxm  # noqa: E402


FULL_SHAPE = (128, 256, 256)
LOW_SHAPE = (64, 128, 128)
NB_FEATURES = [[32, 64, 64, 64, 64], [64, 64, 64, 64, 64, 32, 16, 16]]


class Haar3DAnalysisOnly(nn.Module):
    """The same trailing-pad Haar analysis used by 256_128model_Train-Copy1."""

    def __init__(self):
        super().__init__()
        low = torch.tensor([1.0, 1.0], dtype=torch.float32) / math.sqrt(2.0)
        high = torch.tensor([1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)
        filters = []
        for z in (low, high):
            for y in (low, high):
                for x in (low, high):
                    filters.append(z[:, None, None] * y[None, :, None] * x[None, None, :])
        self.register_buffer("weight", torch.stack(filters).unsqueeze(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.conv3d(F.pad(image, (0, 1, 0, 1, 0, 1)), self.weight)


def downsample(bands: torch.Tensor) -> torch.Tensor:
    return bands[:, :, ::2, ::2, ::2]


def upsample(bands: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        bands.shape[0], bands.shape[1], bands.shape[2] * 2, bands.shape[3] * 2, bands.shape[4] * 2,
        dtype=bands.dtype, device=bands.device,
    )
    output[:, :, ::2, ::2, ::2] = bands
    return output


def synthesis_filters(device: torch.device) -> torch.Tensor:
    low = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device) / math.sqrt(2.0)
    filters = []
    for z in (low, high):
        for y in (low, high):
            for x in (low, high):
                filters.append(z[:, None, None] * y[None, :, None] * x[None, None, :])
    return torch.flip(torch.stack(filters), dims=[1, 2, 3]).unsqueeze(1)


def synthesis(bands: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    d, h, w = bands.shape[2:]
    pieces = []
    for band_index in range(bands.shape[1]):
        piece = F.conv3d(bands[:, band_index : band_index + 1], filters[band_index : band_index + 1], padding=1)
        pieces.append(piece[:, :, :d, :h, :w])
    return torch.cat(pieces, dim=1).sum(dim=1, keepdim=True)


def metrics(reference: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    difference = prediction - reference
    return {
        "mae": float(difference.abs().mean()),
        "mse": float(difference.square().mean()),
        "max_abs_error": float(difference.abs().max()),
    }


class VolumeStore:
    """Read one volume at a time without holding the full archive in RAM."""

    def __init__(self, raw: np.ndarray):
        if raw.ndim != 4:
            raise ValueError(f"Expected a 4-D Train array, got {raw.shape}")
        self.raw = raw
        if raw.shape[:3] == FULL_SHAPE:
            self.sample_axis = 3
            self.count = raw.shape[3]
        elif raw.shape[1:] == FULL_SHAPE:
            self.sample_axis = 0
            self.count = raw.shape[0]
        else:
            raise ValueError(f"Cannot normalize Train array {raw.shape} to samples of {FULL_SHAPE}")

    def get(self, index: int) -> np.ndarray:
        index %= self.count
        source = self.raw[..., index] if self.sample_axis == 3 else self.raw[index]
        return np.ascontiguousarray(source, dtype=np.float32)[None]


def load_volumes(data_path: Path, cache_path: Path) -> VolumeStore:
    """Use a disk-backed NPY cache because ``np.load(npz)['Train']`` needs ~25 GiB RAM.

    The initial extraction is streamed, so it does not allocate the full array.
    It needs approximately the uncompressed array size as free disk space once.
    """
    if data_path.suffix.lower() == ".npy":
        return VolumeStore(np.load(data_path, mmap_mode="r"))
    if data_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected .npz or .npy data, got {data_path}")
    if not cache_path.exists():
        print(f"Creating disk-backed cache (one-time, no 25 GiB RAM allocation): {cache_path}")
        with zipfile.ZipFile(data_path) as archive:
            member = next((name for name in archive.namelist() if Path(name).name == "Train.npy"), None)
            if member is None:
                raise KeyError(f"{data_path} has no Train.npy member: {archive.namelist()}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, cache_path.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    return VolumeStore(np.load(cache_path, mmap_mode="r"))


def smooth_random_full_flow(seed: int, shift_range: float, device: torch.device) -> torch.Tensor:
    """Exact pretraining policy: coarse U(-A,A), SciPy Gaussian smoothing, then interpolate."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coarse = (torch.rand((1, 3, 8, 16, 16), generator=generator) * 2.0 - 1.0) * shift_range
    smoothed = gaussian_filter(coarse.numpy(), sigma=[0, 0, 2.0, 2.0, 2.0])
    return F.interpolate(torch.from_numpy(smoothed).to(device=device, dtype=torch.float32), size=FULL_SHAPE,
                         mode="trilinear", align_corners=False)


def prepare_wavelet_pipeline(device: torch.device):
    analysis = Haar3DAnalysisOnly().to(device).eval()
    return analysis, synthesis_filters(device), vxm.layers.SpatialTransformer(FULL_SHAPE).to(device), vxm.layers.SpatialTransformer(LOW_SHAPE).to(device)


def warp_in_wavelet_domain(image: torch.Tensor, low_flow: torch.Tensor, analysis, filters, low_transformer) -> torch.Tensor:
    bands = downsample(analysis(image))
    warped = torch.cat(
        [low_transformer(bands[:, index : index + 1], low_flow) for index in range(bands.shape[1])], dim=1
    )
    return synthesis(upsample(warped), filters)


def make_model(device: torch.device, checkpoint: Path | None) -> nn.Module:
    model = vxm.networks.VxmDense_128_256_256(FULL_SHAPE, NB_FEATURES, int_steps=0).to(device)
    if checkpoint is not None:
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location=device)
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model.load_state_dict(state, strict=True)
    return model


def save_constant_shift_figure(output: Path, moving: torch.Tensor, full_warp: torch.Tensor, low_warp: torch.Tensor, shift: int):
    index = moving.shape[2] // 2
    difference = (low_warp - full_warp)[0, 0, index].detach().cpu().numpy()
    values = [item[0, 0, index].detach().cpu().numpy() for item in (moving, full_warp, low_warp)]
    figure, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for axis, value, title in zip(axes[:3], values, ["Moving", f"Full warp: x={shift}", f"Band warp: x={shift // 2}"]):
        axis.imshow(value, cmap="gray"); axis.set_title(title); axis.axis("off")
    image = axes[3].imshow(difference, cmap="coolwarm")
    axes[3].set_title("Band − full"); axes[3].axis("off"); figure.colorbar(image, ax=axes[3], shrink=0.8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def experiment_a(moving, analysis, filters, full_transformer, low_transformer, output_dir: Path) -> list[dict]:
    """Compare full x translations with the physically corresponding half-grid translations."""
    rows = []
    for full_shift in (1, 2):
        full_flow = torch.zeros((1, 3, *FULL_SHAPE), dtype=moving.dtype, device=moving.device)
        full_flow[:, 2] = float(full_shift)  # channel 2 is x in this repository.
        full_warp = full_transformer(moving, full_flow)
        low_flow = F.interpolate(full_flow, size=LOW_SHAPE, mode="trilinear", align_corners=False) * 0.5
        band_warp = warp_in_wavelet_domain(moving, low_flow, analysis, filters, low_transformer)
        row = {"full_x_shift_voxels": full_shift, "low_x_shift_voxels": full_shift / 2.0}
        row.update(metrics(full_warp, band_warp))
        rows.append(row)
        save_constant_shift_figure(output_dir / f"experiment_A_x{full_shift}.png", moving, full_warp, band_warp, full_shift)
    return rows


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def parameter_update_metrics(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float | int]:
    differences = [
        (parameter.detach() - before[name]).abs().reshape(-1)
        for name, parameter in model.named_parameters()
    ]
    merged = torch.cat(differences)
    changed = sum(bool(difference.max().item() > 0.0) for difference in differences)
    return {"parameter_max_abs_change": float(merged.max()), "parameter_mean_abs_change": float(merged.mean()),
            "updated_parameter_tensors": changed, "total_parameter_tensors": len(differences)}


def gradient_metrics(model: nn.Module) -> dict[str, float | int]:
    norms = [parameter.grad.detach().norm() for parameter in model.parameters() if parameter.grad is not None]
    if not norms:
        return {"gradient_l2_norm": 0.0, "parameters_with_gradient": 0}
    return {"gradient_l2_norm": float(torch.linalg.vector_norm(torch.stack(norms))), "parameters_with_gradient": len(norms)}


def train_scale_condition(volumes, scale: float, steps: int, seed: int, shift_range: float, device: torch.device,
                          checkpoint: Path | None, analysis, filters, full_transformer, low_transformer):
    """Experiment B/C: one identical synthetic sequence for one label-scale condition."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(device, checkpoint).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rows = []
    diagnostic = None

    for step in range(steps):
        # One fixed source image keeps the two label-scale conditions exactly comparable
        # and avoids loading the large NPZ array into RAM.
        moving_np = volumes.get(0)
        moving = torch.from_numpy(moving_np).unsqueeze(1).to(device)
        full_teacher = smooth_random_full_flow(seed + step, shift_range, device)
        target = full_transformer(moving, full_teacher)
        moving_bands = downsample(analysis(moving))
        target_bands = downsample(analysis(target))
        # interpolate changes only sampling locations.  ``scale`` is the experimental magnitude conversion.
        low_teacher = F.interpolate(full_teacher, size=LOW_SHAPE, mode="trilinear", align_corners=False) * scale

        before = parameter_snapshot(model) if step == 0 else None
        optimizer.zero_grad(set_to_none=True)
        prediction = model(moving_bands, target_bands)
        moved = warp_in_wavelet_domain(moving, prediction, analysis, filters, low_transformer)
        raw_vec = F.mse_loss(prediction, low_teacher)
        raw_image = F.mse_loss(moved, target)
        total = raw_vec * 0.01 + raw_image * 100.0
        total.backward()
        grad = gradient_metrics(model)
        optimizer.step()

        row = {
            "step": step + 1, "teacher_scale": scale,
            "loss_vec_raw": float(raw_vec.detach()), "loss_image_raw": float(raw_image.detach()),
            "loss_vec_weighted": float((raw_vec * 0.01).detach()),
            "loss_image_weighted": float((raw_image * 100.0).detach()), "total_loss": float(total.detach()),
            "prediction_mean": float(prediction.detach().mean()), "prediction_std": float(prediction.detach().std()),
            "prediction_max_abs": float(prediction.detach().abs().max()),
            "teacher_prediction_mae": float((prediction.detach() - low_teacher).abs().mean()),
            **grad,
        }
        if before is not None:
            row.update(parameter_update_metrics(model, before))
        rows.append(row)
        diagnostic = {
            "moving": moving.detach(), "target": target.detach(), "moved": moved.detach(),
            "teacher": low_teacher.detach(), "prediction": prediction.detach(),
        }
    return rows, model, diagnostic


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def plot_experiment_b(rows_by_scale: dict[float, list[dict]], output: Path):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [("total_loss", "Total loss"), ("loss_vec_raw", "Raw DVF MSE"),
             ("loss_image_raw", "Raw image MSE"), ("teacher_prediction_mae", "DVF MAE")]
    for axis, (column, title) in zip(axes.flat, plots):
        for scale, rows in rows_by_scale.items():
            axis.plot([row["step"] for row in rows], [row[column] for row in rows], label=f"teacher × {scale:g}")
        axis.set(title=title, xlabel="step"); axis.grid(alpha=0.25); axis.legend()
    figure.savefig(output, dpi=180); plt.close(figure)


def save_experiment_b_diagnostic(output: Path, diagnostic: dict[str, torch.Tensor], scale: float):
    """Save the requested image and DVF comparison for each label-scale condition."""
    moving, target, moved = [diagnostic[name][0, 0] for name in ("moving", "target", "moved")]
    teacher_x, predicted_x = [diagnostic[name][0, 2] for name in ("teacher", "prediction")]
    z_image, z_flow = moving.shape[0] // 2, teacher_x.shape[0] // 2
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    images = [moving[z_image], target[z_image], moved[z_image], moved[z_image] - target[z_image]]
    titles = ["Moving", "Moving′ target", "Moved", "Moved − target"]
    for axis, image, title in zip(axes[0], images, titles):
        shown = axis.imshow(image.detach().cpu(), cmap="coolwarm" if "−" in title else "gray")
        axis.set_title(title); axis.axis("off")
        if "−" in title:
            figure.colorbar(shown, ax=axis, shrink=0.75)
    flows = [teacher_x[z_flow], predicted_x[z_flow], predicted_x[z_flow] - teacher_x[z_flow]]
    flow_titles = ["Teacher x-flow", "Predicted x-flow", "Prediction − teacher"]
    for axis, image, title in zip(axes[1, :3], flows, flow_titles):
        shown = axis.imshow(image.detach().cpu(), cmap="coolwarm")
        axis.set_title(title); axis.axis("off"); figure.colorbar(shown, ax=axis, shrink=0.75)
    axes[1, 3].axis("off")
    figure.suptitle(f"Experiment B: low-resolution teacher scale = {scale:g}")
    figure.savefig(output, dpi=180); plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=SAITO_ROOT / "Data" / "TrainData_NoBed.npz")
    parser.add_argument("--checkpoint", type=Path, default=SAITO_ROOT / "model_analysis_pipeline_pretrain.pth")
    parser.add_argument("--steps", type=int, default=100, help="Use 100 first; 1000 is the longer requested comparison.")
    parser.add_argument("--shift-range", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        raise ValueError("--steps must be between 1 and 1000")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = SAITO_ROOT / "dvf_wavelet_verification"
    output_dir.mkdir(exist_ok=True)
    volumes = load_volumes(args.data, output_dir / "TrainData_NoBed_Train.npy")
    moving = torch.from_numpy(volumes.get(0)).unsqueeze(1).to(device)
    analysis, filters, full_transformer, low_transformer = prepare_wavelet_pipeline(device)

    a_rows = experiment_a(moving, analysis, filters, full_transformer, low_transformer, output_dir)
    write_csv(output_dir / "experiment_A_constant_translation.csv", a_rows)

    checkpoint = args.checkpoint if args.checkpoint.exists() else None
    if checkpoint is None:
        print(f"WARNING: checkpoint not found; B/C use identical random initialization: {args.checkpoint}")
    b_rows, b_diagnostics = {}, {}
    for scale in (1.0, 0.5):
        rows, _, diagnostic = train_scale_condition(volumes, scale, args.steps, args.seed, args.shift_range, device, checkpoint,
                                                    analysis, filters, full_transformer, low_transformer)
        b_rows[scale] = rows
        b_diagnostics[scale] = diagnostic
        write_csv(output_dir / f"experiment_BC_teacher_scale_{scale:g}.csv", rows)
        save_experiment_b_diagnostic(output_dir / f"experiment_B_teacher_scale_{scale:g}.png", diagnostic, scale)
    plot_experiment_b(b_rows, output_dir / "experiment_B_teacher_scale_comparison.png")

    summary = {
        "device": str(device), "data": str(args.data), "checkpoint": str(checkpoint) if checkpoint else None,
        "source_volume_count": volumes.count,
        "steps": args.steps, "shift_range": args.shift_range, "experiment_A": a_rows,
        "experiment_C_first_step": {str(scale): rows[0] for scale, rows in b_rows.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved verification results: {output_dir}")


if __name__ == "__main__":
    main()
