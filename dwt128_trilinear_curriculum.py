"""Curriculum learning for the 128-resolution trilinear + 3-D Haar model.

Pipeline
--------
raw CT (128, 256, 256) -> trilinear (64, 128, 128) -> 8 Haar bands
(32, 64, 64) -> four internally grouped bands for DVF estimation -> warp all
8 bands -> inverse Haar -> reconstructed image (64, 128, 128).

The curriculum displacement is expressed in pixels of the reconstructed
``(64, 128, 128)`` image. Since the Haar coefficient grid is half-sized in
every axis, the teacher DVF is divided by two before comparing it with the
predicted coefficient-grid DVF.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from voxelmorph.torch.external_wavelet_registration import ExternalWaveletVxm
from voxelmorph.torch.layers import SpatialTransformer


TARGET_SHAPE = (64, 128, 128)
COEFFICIENT_SHAPE = (32, 64, 64)
SOURCE_SHAPE = (128, 256, 256)


def to_n_dhw(array: np.ndarray, label: str = 'volume') -> np.ndarray:
    """Convert a supported 4-D archive layout to ``(N, D, H, W)``."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f'{label} must be 4-D, got {array.shape}')
    if array.shape[1:] in (SOURCE_SHAPE, TARGET_SHAPE):
        return array
    if array.shape[:3] in (SOURCE_SHAPE, TARGET_SHAPE):
        return np.transpose(array, (3, 0, 1, 2))
    raise ValueError(
        f'{label} has unsupported shape {array.shape}; expected N/D/H/W or '
        f'D/H/W/N with spatial size {SOURCE_SHAPE} or {TARGET_SHAPE}.'
    )


def load_training_volumes(data_path: str | Path, key: str = 'Train') -> np.ndarray:
    """Load training CT volumes without changing their original resolution."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f'Training archive was not found: {data_path.resolve()}')
    with np.load(data_path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f'{data_path.name} has no {key!r}; available keys: {archive.files}')
        return to_n_dhw(archive[key], label=key)


def resize_to_target(volumes: torch.Tensor) -> torch.Tensor:
    """Apply the same 3-linear preprocessing as the ordinary 128 model."""
    return F.interpolate(volumes, size=TARGET_SHAPE, mode='trilinear', align_corners=False)


def haar_analysis_3d(x: torch.Tensor) -> torch.Tensor:
    """One-level orthonormal Haar DWT: target image -> 8 coefficient bands."""
    if tuple(x.shape[2:]) != TARGET_SHAPE:
        raise ValueError(f'expected image size {TARGET_SHAPE}, got {tuple(x.shape[2:])}')
    low = torch.tensor([1.0, 1.0], dtype=x.dtype, device=x.device) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], dtype=x.dtype, device=x.device) / math.sqrt(2.0)
    filters = torch.stack([
        torch.einsum('i,j,k->ijk', z, y, x_axis)
        for z in (low, high)
        for y in (low, high)
        for x_axis in (low, high)
    ]).unsqueeze(1)
    return F.conv3d(x, filters, stride=2)


def make_model(device: torch.device, feature_channels: int = 16) -> ExternalWaveletVxm:
    """The existing network groups 8 bands into 4 groups internally."""
    return ExternalWaveletVxm(
        coefficient_shape=COEFFICIENT_SHAPE,
        feature_channels=feature_channels,
    ).to(device)


def smooth_random_target_flow(batch_size: int, max_shift: float, device: torch.device) -> torch.Tensor:
    """Create a smooth synthetic DVF in reconstructed-image pixel units."""
    coarse = (torch.rand(batch_size, 3, 8, 8, 8, device=device) * 2.0 - 1.0) * max_shift
    return F.interpolate(coarse, size=TARGET_SHAPE, mode='trilinear', align_corners=False)


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dz = (flow[:, :, 1:] - flow[:, :, :-1]).square().mean()
    dy = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square().mean()
    dx = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).square().mean()
    return (dz + dy + dx) / 3.0


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Initial checkpoint was not found: {checkpoint_path.resolve()}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def _write_history(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_curriculum_pretraining(
    data_path: str | Path = 'Data/TrainData_NoBed.npz',
    output_dir: str | Path = '128dwt_trilinear_curriculum_checkpoints',
    *,
    total_epochs: int = 80_000,
    stage_epochs: int = 2_000,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    image_weight: float = 100.0,
    flow_weight: float = 0.01,
    smoothness_weight: float = 0.1,
    feature_channels: int = 16,
    initial_checkpoint: str | Path | None = None,
    checkpoint_prefix: str = '128dwt_trilinear_curriculum',
    seed: int = 20260826,
    device: torch.device | None = None,
):
    """Run the ±1 to ±40 px curriculum and save every 2,000-epoch stage."""
    if total_epochs % stage_epochs:
        raise ValueError('total_epochs must be divisible by stage_epochs')
    if total_epochs // stage_epochs != 40:
        raise ValueError('This 1→40 px curriculum requires 80,000 / 2,000 epochs.')

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    volumes = load_training_volumes(data_path)
    if len(volumes) == 0:
        raise ValueError('The training archive contains no volumes.')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(device, feature_channels=feature_channels)
    if initial_checkpoint is not None:
        _load_model_weights(model, initial_checkpoint, device)
        print(f'Loaded initial weights: {Path(initial_checkpoint).resolve()}')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    target_transformer = SpatialTransformer(TARGET_SHAPE).to(device)
    rng = np.random.default_rng(seed)
    history: list[dict] = []

    for epoch in tqdm(range(1, total_epochs + 1), desc='128-DWT curriculum'):
        stage = (epoch - 1) // stage_epochs + 1
        max_shift = float(stage)
        indices = rng.integers(0, len(volumes), size=batch_size)
        moving_source = torch.from_numpy(volumes[indices]).unsqueeze(1).to(device)
        moving_target = resize_to_target(moving_source)

        teacher_target_flow = smooth_random_target_flow(batch_size, max_shift, device)
        fixed_target = target_transformer(moving_target, teacher_target_flow)
        moving_coefficients = haar_analysis_3d(moving_target)
        fixed_coefficients = haar_analysis_3d(fixed_target)
        moved_target, _, predicted_coefficient_flow = model(moving_coefficients, fixed_coefficients)

        # One coefficient-grid pixel equals 2 px in the reconstructed image.
        teacher_coefficient_flow = F.interpolate(
            teacher_target_flow, size=COEFFICIENT_SHAPE, mode='trilinear', align_corners=False
        ) / 2.0
        image_loss = F.mse_loss(moved_target, fixed_target)
        flow_loss = F.mse_loss(predicted_coefficient_flow, teacher_coefficient_flow)
        smooth_loss = flow_smoothness(predicted_coefficient_flow)
        loss = image_weight * image_loss + flow_weight * flow_loss + smoothness_weight * smooth_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f'epoch={epoch:05d}/{total_epochs}, stage={stage:02d}, ±{max_shift:.0f}px, '
                f'loss={loss.item():.6f}, image={image_loss.item():.6f}, '
                f'dvf={flow_loss.item():.6f}, smooth={smooth_loss.item():.6f}'
            )

        if epoch % stage_epochs == 0:
            row = {
                'epoch': epoch,
                'stage': stage,
                'max_shift_target_pixels': max_shift,
                'loss': loss.item(),
                'image_loss': image_loss.item(),
                'flow_loss': flow_loss.item(),
                'smoothness_loss': smooth_loss.item(),
            }
            history.append(row)
            checkpoint_path = output_dir / f'{checkpoint_prefix}_stage_{stage:02d}_epoch_{epoch:05d}.pth'
            torch.save({
                **row,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'target_shape': TARGET_SHAPE,
                'coefficient_shape': COEFFICIENT_SHAPE,
                'feature_channels': feature_channels,
            }, checkpoint_path)
            _write_history(history, output_dir / f'{checkpoint_prefix}_history.csv')
            print(f'Saved: {checkpoint_path.resolve()}')

    final_path = output_dir / f'{checkpoint_prefix}_1to40_final.pth'
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'target_shape': TARGET_SHAPE,
        'coefficient_shape': COEFFICIENT_SHAPE,
        'feature_channels': feature_channels,
        'initial_checkpoint': str(initial_checkpoint) if initial_checkpoint else None,
    }, final_path)
    print(f'Final checkpoint: {final_path.resolve()}')
    return model, final_path, history
