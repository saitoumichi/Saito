"""Small, staged checks for the 256→128 Wavelet DVF pipeline.

This file never trains for 80,000 epochs and never writes a model checkpoint.
It is intended for a lower-memory PC before the full re-training.

Examples (run in the Saito folder):
  # A. No CT data and no neural network: verify the x=2 -> x=1 conversion.
  python low_memory_wavelet_check.py --stage A

  # B. One individual 3-D CT file only: make the same constant-shift figure.
  python low_memory_wavelet_check.py --stage B --image Data/Longitudinal22/pair1/registered_masked_A_xxx.npz

  # C. One volume only, 10 learning steps: inspect raw and weighted losses.
  python low_memory_wavelet_check.py --stage C --image one_volume.npy --steps 10

Do not pass Data/TrainData_NoBed.npz. Its ``Train`` array requires tens of GB
to decompress.  For stage B/C, use an NPZ/NPY which contains one CT volume.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


SAITO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SAITO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import voxelmorph as vxm  # noqa: E402


FULL_SHAPE = (128, 256, 256)
LOW_SHAPE = (64, 128, 128)
NB_FEATURES = [[32, 64, 64, 64, 64], [64, 64, 64, 64, 64, 32, 16, 16]]


class HaarAnalysis(nn.Module):
    """Same trailing-pad Haar analysis as 256_128model_Train-Copy1."""

    def __init__(self):
        super().__init__()
        low = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
        high = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
        kernels = [z[:, None, None] * y[None, :, None] * x[None, None, :]
                   for z in (low, high) for y in (low, high) for x in (low, high)]
        self.register_buffer("weight", torch.stack(kernels).unsqueeze(1))

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


def make_synthesis_filters(device: torch.device) -> torch.Tensor:
    low = torch.tensor([1.0, 1.0], device=device) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], device=device) / math.sqrt(2.0)
    kernels = [z[:, None, None] * y[None, :, None] * x[None, None, :]
               for z in (low, high) for y in (low, high) for x in (low, high)]
    return torch.flip(torch.stack(kernels), dims=[1, 2, 3]).unsqueeze(1)


def synthesis(bands: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    depth, height, width = bands.shape[2:]
    pieces = []
    for index in range(8):
        piece = F.conv3d(bands[:, index:index + 1], filters[index:index + 1], padding=1)
        pieces.append(piece[:, :, :depth, :height, :width])
    return torch.cat(pieces, dim=1).sum(dim=1, keepdim=True)


def wavelet_warp(image, low_flow, analysis, filters, low_transformer):
    bands = downsample(analysis(image))
    warped = torch.cat([low_transformer(bands[:, i:i + 1], low_flow) for i in range(8)], dim=1)
    return synthesis(upsample(warped), filters)


def make_test_image(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Small synthetic volume with asymmetric features; requires no CT data."""
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, shape[0], device=device),
        torch.linspace(-1, 1, shape[1], device=device),
        torch.linspace(-1, 1, shape[2], device=device), indexing="ij",
    )
    image = torch.exp(-((x + .28) ** 2 + (y - .12) ** 2 + z ** 2) * 10)
    image += .7 * torch.exp(-((x - .35) ** 2 + (y + .25) ** 2 + (z + .18) ** 2) * 30)
    return image[None, None]


def read_one_volume(path: Path, key: str | None, index: int, device: torch.device) -> torch.Tensor:
    """Read one volume, explicitly refusing the large multi-volume Train archive."""
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r")
    elif path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        selected_key = key or (archive.files[0] if len(archive.files) == 1 else None)
        if selected_key is None:
            raise ValueError(f"Specify --key. Available keys: {archive.files}")
        # Longitudinal22の個別患者NPZもキー名は ``Train`` なので、
        # キー名ではなく全症例アーカイブのファイル名で判定する。
        if selected_key == "Train" and path.name == "TrainData_NoBed.npz":
            raise ValueError("TrainData_NoBed.npz は使えません。1症例だけのNPY/NPZを指定してください。")
        array = archive[selected_key]
    else:
        raise ValueError("--image must be a .npy or .npz file")

    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 4:
        # Supports either (N,D,H,W) or (D,H,W,N), but reads only one selected volume.
        if tuple(array.shape[1:]) == FULL_SHAPE:
            array = array[index % array.shape[0]]
        elif tuple(array.shape[:3]) == FULL_SHAPE:
            array = array[..., index % array.shape[3]]
        else:
            raise ValueError(f"Cannot identify the sample axis in {array.shape}")
    if array.ndim != 3:
        raise ValueError(f"Expected a single 3-D volume, got {array.shape}")
    if any(length % 2 for length in array.shape):
        raise ValueError(f"All dimensions must be even for this Haar test, got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None].to(device)


def metric_dict(reference: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    error = prediction - reference
    return {"mae": float(error.abs().mean()), "mse": float(error.square().mean()),
            "max_abs_error": float(error.abs().max())}


def save_shift_figure(output: Path, moving, full_warp, band_warp, full_shift: int):
    mid = moving.shape[2] // 2
    values = [value[0, 0, mid].detach().cpu().numpy() for value in (moving, full_warp, band_warp)]
    difference = (band_warp - full_warp)[0, 0, mid].detach().cpu().numpy()
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, value, title in zip(axes[:3], values, ["Moving", f"Full x={full_shift}", f"Wavelet x={full_shift / 2}"]):
        axis.imshow(value, cmap="gray"); axis.set_title(title); axis.axis("off")
    handle = axes[3].imshow(difference, cmap="coolwarm")
    axes[3].set_title("Wavelet − full"); axes[3].axis("off")
    figure.colorbar(handle, ax=axes[3], shrink=.8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def constant_shift_check(moving, output_dir: Path, device: torch.device):
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(moving.shape[2:])
    low_shape = tuple(length // 2 for length in shape)
    analysis = HaarAnalysis().to(device).eval()
    filters = make_synthesis_filters(device)
    full_transformer = vxm.layers.SpatialTransformer(shape).to(device)
    low_transformer = vxm.layers.SpatialTransformer(low_shape).to(device)
    rows = []
    for full_shift in (1, 2):
        full_flow = torch.zeros((1, 3, *shape), device=device)
        full_flow[:, 2] = full_shift  # In this repository flow channel 2 is x/W.
        full_warp = full_transformer(moving, full_flow)
        # This is the hypothesis being tested: decimation by 2 means low-grid flow is half.
        low_flow = F.interpolate(full_flow, size=low_shape, mode="trilinear", align_corners=False) * .5
        band_warp = wavelet_warp(moving, low_flow, analysis, filters, low_transformer)
        row = {"full_x_shift_voxels": full_shift, "low_x_shift_voxels": full_shift / 2}
        row.update(metric_dict(full_warp, band_warp))
        rows.append(row)
        save_shift_figure(output_dir / f"constant_shift_x{full_shift}.png", moving, full_warp, band_warp, full_shift)
    with (output_dir / "constant_shift_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print("\nConstant-shift check")
    for row in rows:
        print(row)


def smooth_teacher(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = (torch.rand((1, 3, 8, 16, 16), generator=generator) * 2 - 1) * 2.0
    smooth = gaussian_filter(coarse.numpy(), sigma=[0, 0, 2, 2, 2])
    return F.interpolate(torch.from_numpy(smooth).float().to(device), size=FULL_SHAPE,
                         mode="trilinear", align_corners=False)


def short_training_check(moving, steps: int, output_dir: Path, device: torch.device, checkpoint: Path | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if tuple(moving.shape[2:]) != FULL_SHAPE:
        raise ValueError(f"Stage C requires {FULL_SHAPE}, got {tuple(moving.shape[2:])}")
    analysis = HaarAnalysis().to(device).eval()
    filters = make_synthesis_filters(device)
    full_transformer = vxm.layers.SpatialTransformer(FULL_SHAPE).to(device)
    low_transformer = vxm.layers.SpatialTransformer(LOW_SHAPE).to(device)
    model = vxm.networks.VxmDense_128_256_256(FULL_SHAPE, NB_FEATURES, int_steps=0).to(device)
    if checkpoint:
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rows = []
    before = [parameter.detach().clone() for parameter in model.parameters()]
    for step in range(steps):
        teacher_full = smooth_teacher(1234 + step, device)
        moving_prime = full_transformer(moving, teacher_full)
        moving_w = downsample(analysis(moving))
        fixed_w = downsample(analysis(moving_prime))
        teacher_low = F.interpolate(teacher_full, size=LOW_SHAPE, mode="trilinear", align_corners=False) * .5
        optimizer.zero_grad(set_to_none=True)
        vec = model(moving_w, fixed_w)
        transformed = wavelet_warp(moving, vec, analysis, filters, low_transformer)
        dvf_raw = F.mse_loss(teacher_low, vec)
        image_raw = F.mse_loss(moving_prime, transformed)
        dvf_weighted, image_weighted = dvf_raw * .01, image_raw * 100.0
        total = dvf_weighted + image_weighted
        total.backward()
        gradient_norm = float(torch.linalg.vector_norm(torch.stack(
            [parameter.grad.detach().norm() for parameter in model.parameters() if parameter.grad is not None]
        )))
        optimizer.step()
        ratio = float(image_weighted.detach() / dvf_weighted.detach().clamp_min(1e-12))
        row = {"step": step + 1, "dvf_raw": float(dvf_raw), "image_raw": float(image_raw),
               "dvf_weighted": float(dvf_weighted), "image_weighted": float(image_weighted),
               "total": float(total), "image_to_dvf": ratio, "gradient_l2_norm": gradient_norm,
               "vec_mean": float(vec.detach().mean()), "vec_std": float(vec.detach().std()),
               "vec_abs_max": float(vec.detach().abs().max())}
        rows.append(row)
        print(f"step {step + 1}/{steps}: DVF={row['dvf_raw']:.5g}×.01={row['dvf_weighted']:.5g}, "
              f"image={row['image_raw']:.5g}×100={row['image_weighted']:.5g}, image/DVF={ratio:.2f}")
    changes = torch.cat([(parameter.detach() - old).abs().reshape(-1) for parameter, old in zip(model.parameters(), before)])
    print(f"Parameter update: max={changes.max().item():.6g}, mean={changes.mean().item():.6g}")
    with (output_dir / "short_training_loss_balance.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A", "B", "C"), required=True)
    parser.add_argument("--image", type=Path, help="One-volume .npy/.npz file (required for B/C).")
    parser.add_argument("--key", help="NPZ key, when the one-volume archive has multiple keys.")
    parser.add_argument("--index", type=int, default=0, help="Volume index only for a small 4-D input.")
    parser.add_argument("--steps", type=int, default=10, help="Stage C training steps (start with 10).")
    parser.add_argument("--checkpoint", type=Path, help="Optional pretraining state_dict for Stage C; never modified.")
    parser.add_argument("--output", type=Path, default=Path("low_memory_wavelet_check"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; outputs: {args.output.resolve()}")
    if args.stage == "A":
        moving = make_test_image((32, 64, 64), device)
        constant_shift_check(moving, args.output, device)
    else:
        if args.image is None:
            parser.error("--image is required for stage B and C")
        moving = read_one_volume(args.image, args.key, args.index, device)
        print(f"Loaded one volume: {tuple(moving.shape)}")
        if args.stage == "B":
            constant_shift_check(moving, args.output, device)
        else:
            short_training_check(moving, args.steps, args.output, device, args.checkpoint)


if __name__ == "__main__":
    main()
