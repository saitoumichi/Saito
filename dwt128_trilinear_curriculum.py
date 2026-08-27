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
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from voxelmorph.torch.external_wavelet_registration import ExternalWaveletVxm
from voxelmorph.torch.layers import SpatialTransformer


TARGET_SHAPE = (64, 128, 128)
COEFFICIENT_SHAPE = (32, 64, 64)
SOURCE_SHAPE = (128, 256, 256)


def to_n_dhw(array: np.ndarray, label: str = 'volume') -> np.ndarray:
    """Convert a supported 4-D archive layout to ``(N, D, H, W)``."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        if array.shape == SOURCE_SHAPE:
            return array[np.newaxis, ...]
        if array.shape == (256, 256, 128):
            return np.transpose(array, (2, 0, 1))[np.newaxis, ...]
    if array.ndim != 4:
        raise ValueError(f'{label} must be 3-D/4-D, got {array.shape}')
    if array.shape[1:] in (SOURCE_SHAPE, TARGET_SHAPE):
        return array
    if array.shape[:3] in (SOURCE_SHAPE, TARGET_SHAPE):
        return np.transpose(array, (3, 0, 1, 2))
    if array.shape[1:] == (256, 256, 128):
        return np.transpose(array, (0, 3, 1, 2))
    if array.shape[:3] == (256, 256, 128):
        return np.transpose(array, (3, 2, 0, 1))
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


def load_different_patient_volumes(data_root: str | Path, key: str = 'Train') -> tuple[np.ndarray, list[str]]:
    """Load both volumes from every TestData pair folder for fine-tuning.

    The source images are used with synthetic DVFs, exactly as in the
    256→128 different-patient fine-tuning notebook.  This avoids treating the
    held-out fixed image as a deformation ground truth.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f'Different-patient data directory was not found: {data_root.resolve()}')
    pair_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not pair_dirs:
        raise FileNotFoundError(f'No pair folders were found under: {data_root.resolve()}')

    arrays: list[np.ndarray] = []
    source_files: list[str] = []
    for pair_dir in pair_dirs:
        npz_paths = sorted(pair_dir.glob('*.npz'))
        if len(npz_paths) != 2:
            raise ValueError(f'{pair_dir}: expected exactly two .npz files, found {len(npz_paths)}')
        for npz_path in npz_paths:
            arrays.append(load_training_volumes(npz_path, key=key))
            source_files.append(str(npz_path.resolve()))

    volumes = np.concatenate(arrays, axis=0)
    if len(volumes) == 0:
        raise ValueError('Different-patient data contains no volumes.')
    return volumes, source_files


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


def _save_loss_plot(rows: list[dict], path: Path, *, show_in_notebook: bool = True) -> None:
    """Save and, in Jupyter, display the loss chart accumulated so far."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    epochs = [row['epoch'] for row in rows]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(epochs, [row['loss'] for row in rows], label='total loss', linewidth=1.4)
    axis.plot(epochs, [row['image_loss'] for row in rows], label='image loss', linewidth=1.0)
    axis.plot(epochs, [row['flow_loss'] for row in rows], label='DVF loss', linewidth=1.0)
    axis.plot(epochs, [row['smoothness_loss'] for row in rows], label='smoothness loss', linewidth=1.0)
    axis.set(title='128-DWT Curriculum Training Loss', xlabel='Epoch', ylabel='Loss')
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show_in_notebook:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass
    plt.close(fig)


def _save_registration_preview(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    moved: torch.Tensor,
    teacher_flow: torch.Tensor,
    predicted_flow: torch.Tensor,
    path: Path,
    epoch: int,
    *,
    show_in_notebook: bool = True,
) -> None:
    """Save a Moving/Fixed/Moved preview and its full-volume metrics."""
    import matplotlib.pyplot as plt

    moving_np = moving[0, 0].detach().float().cpu().numpy()
    fixed_np = fixed[0, 0].detach().float().cpu().numpy()
    moved_np = moved[0, 0].detach().float().cpu().numpy()
    diff_np = np.abs(fixed_np - moved_np)
    mse = float(np.mean((fixed_np - moved_np) ** 2))
    mae = float(np.mean(diff_np))
    fixed_centered = fixed_np - fixed_np.mean()
    moved_centered = moved_np - moved_np.mean()
    ncc = float((fixed_centered * moved_centered).mean() / (
        fixed_centered.std() * moved_centered.std() + 1e-8
    ))
    dvf_mse = float(F.mse_loss(predicted_flow, teacher_flow).detach().cpu())
    slice_index = fixed_np.shape[0] // 2
    vmin, vmax = np.percentile(np.concatenate((moving_np.ravel(), fixed_np.ravel())), (1, 99))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, image, title in zip(
        axes[:3], (moving_np, fixed_np, moved_np), ('Moving', 'Fixed', 'Moved')
    ):
        axis.imshow(image[slice_index], cmap='gray', vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis('off')
    axes[3].imshow(diff_np[slice_index], cmap='magma')
    axes[3].set_title('|Fixed − Moved|')
    axes[3].axis('off')
    fig.suptitle(
        f'Epoch {epoch:,}  |  MSE: {mse:.6f}  MAE: {mae:.6f}  NCC: {ncc:.4f}  DVF MSE: {dvf_mse:.6f}',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if show_in_notebook:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass
    plt.close(fig)


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
            print(
                f'epoch={epoch:05d}/{total_epochs}, stage={stage:02d}, ±{max_shift:.0f}px, '
                f'loss={loss.item():.6f}, image={image_loss.item():.6f}, '
                f'dvf={flow_loss.item():.6f}, smooth={smooth_loss.item():.6f}'
            )

        if epoch % stage_epochs == 0:
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
            _save_loss_plot(history, output_dir / f'{checkpoint_prefix}_loss_progress.png')
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
    _save_loss_plot(history, output_dir / f'{checkpoint_prefix}_loss_progress.png')
    print(f'Final checkpoint: {final_path.resolve()}')
    return model, final_path, history


def run_different_patient_finetuning(
    data_root: str | Path = 'Data/TestData',
    pretrained_checkpoint: str | Path = '128dwt_trilinear_curriculum_checkpoints/128dwt_trilinear_curriculum_1to40_final.pth',
    output_dir: str | Path = '128dwt_trilinear_different_patients_finetune_checkpoints',
    *,
    total_epochs: int = 30_000,
    stage_epochs: int = 2_000,
    batch_size: int = 2,
    learning_rate: float = 1e-6,
    image_weight: float = 100.0,
    flow_weight: float = 0.0,
    smoothness_weight: float = 0.1,
    feature_channels: int = 16,
    checkpoint_prefix: str = '128dwt_trilinear_different_patients_finetune',
    visualization_every: int = 2_000,
    seed: int = 20260826,
    device: torch.device | None = None,
):
    """Fine-tune the 128→64 DWT model for 30k epochs on other-patient CTs."""
    if total_epochs != 30_000 or stage_epochs != 2_000:
        raise ValueError('Different-patient fine-tuning is fixed at 30,000 epochs / 2,000 epochs per stage.')
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    volumes, source_files = load_different_patient_volumes(data_root)
    if len(volumes) < batch_size:
        raise ValueError(f'Only {len(volumes)} volumes were found; batch_size={batch_size} cannot be sampled.')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / 'registration_previews'
    preview_dir.mkdir(exist_ok=True)
    model = make_model(device, feature_channels=feature_channels)
    _load_model_weights(model, pretrained_checkpoint, device)
    print(f'Loaded pretrained weights: {Path(pretrained_checkpoint).resolve()}')
    print(f'Loaded {len(volumes)} volumes from {len(source_files)} files.')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history: list[dict] = []

    for epoch in tqdm(range(1, total_epochs + 1),
                      desc='128→64 image-only different-patient fine-tuning'):

        # moving と fixed を必ず別患者から選ぶ
        moving_indices = rng.integers(0, len(volumes), size=batch_size)
        fixed_indices = rng.integers(0, len(volumes), size=batch_size)

        while np.any(moving_indices == fixed_indices):
            same_patient = moving_indices == fixed_indices
            fixed_indices[same_patient] = rng.integers(
                0, len(volumes), size=same_patient.sum()
            )

        moving_source = torch.from_numpy(
            volumes[moving_indices]
        ).unsqueeze(1).to(device)

        fixed_source = torch.from_numpy(
            volumes[fixed_indices]
        ).unsqueeze(1).to(device)

        # 128×256×256 → 64×128×128
        moving_target = resize_to_target(moving_source)
        fixed_target = resize_to_target(fixed_source)

        moving_coefficients = haar_analysis_3d(moving_target)
        fixed_coefficients = haar_analysis_3d(fixed_target)

        # モデルがDVFを予測してMovingをWarpする
        moved_target, _, predicted_coefficient_flow = model(
            moving_coefficients, fixed_coefficients
        )

        # 教師DVF・DVF誤差は使用しない
        image_loss = F.mse_loss(moved_target, fixed_target)
        smooth_loss = flow_smoothness(predicted_coefficient_flow)
        loss = image_weight * image_loss + smoothness_weight * smooth_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 100 == 0:
            row = {
                'epoch': epoch,
                'loss': loss.item(),
                'image_loss': image_loss.item(),
                'flow_loss': 0.0,  # CSV・グラフ用。学習には使わない
                'smoothness_loss': smooth_loss.item(),
            }
            history.append(row)

            print(
                f'Epoch {epoch:05d}/{total_epochs} | '
                f'loss={loss.item():.6f} | '
                f'image={image_loss.item():.6f} | '
                f'smooth={smooth_loss.item():.6f} | '
                f'patients={moving_indices.tolist()} → {fixed_indices.tolist()}'
            )

        if epoch % stage_epochs == 0:
            checkpoint_path = (
                output_dir
                / f'{checkpoint_prefix}_epoch_{epoch:05d}.pth'
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'target_shape': TARGET_SHAPE,
                'coefficient_shape': COEFFICIENT_SHAPE,
                'feature_channels': feature_channels,
                'pretrained_checkpoint': str(pretrained_checkpoint),
                'data_root': str(data_root),
                'source_files': source_files,
            }, checkpoint_path)

            _write_history(history, output_dir / f'{checkpoint_prefix}_history.csv')
            _save_loss_plot(history, output_dir / f'{checkpoint_prefix}_loss_progress.png')
            print(f'Saved: {checkpoint_path.resolve()}')

        if visualization_every > 0 and epoch % visualization_every == 0:
            slice_index = moving_target.shape[2] // 2

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, image, title in zip(
                axes,
                (
                    moving_target[0, 0, slice_index],
                    fixed_target[0, 0, slice_index],
                    moved_target[0, 0, slice_index],
                ),
                ('Moving', 'Fixed', 'Moved'),
            ):
                axis.imshow(image.detach().cpu(), cmap='gray')
                axis.set_title(title)
                axis.axis('off')

            fig.tight_layout()
            preview_path = preview_dir / f'epoch_{epoch:05d}_registration.png'
            fig.savefig(preview_path, dpi=160, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            print(f'Saved registration preview: {preview_path.resolve()}')
    
    final_path = output_dir / f'{checkpoint_prefix}_final.pth'
    torch.save({
        'epoch': total_epochs, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'target_shape': TARGET_SHAPE,
        'coefficient_shape': COEFFICIENT_SHAPE, 'feature_channels': feature_channels,
        'pretrained_checkpoint': str(pretrained_checkpoint), 'data_root': str(data_root),
        'source_files': source_files,
    }, final_path)
    _write_history(history, output_dir / f'{checkpoint_prefix}_history.csv')
    _save_loss_plot(history, output_dir / f'{checkpoint_prefix}_loss_progress.png')
    print(f'Final checkpoint: {final_path.resolve()}')
    return model, final_path, history
