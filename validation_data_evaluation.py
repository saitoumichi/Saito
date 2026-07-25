"""Fine-tuning validation helpers for lung registration.

Run this file from the notebook *after* `volumes`, `lung_masks`, `device`,
and `reconstruct_warped()` have been defined:

    %run validation_data_evaluation.py

The returned validation patients must be excluded from the fine-tuning
generator.  Do not use final test patients here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def make_train_validation_split(
    patient_count: int,
    validation_fraction: float = 0.10,
    seed: int = 20260725,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible, non-overlapping train and validation patient IDs."""
    if patient_count < 20:
        raise ValueError("At least 20 patients are required for a train/validation split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    patient_ids = np.arange(patient_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    validation_count = max(2, int(round(patient_count * validation_fraction)))
    validation_ids = np.sort(patient_ids[:validation_count])
    train_ids = np.sort(patient_ids[validation_count:])
    return train_ids, validation_ids


def make_fixed_pairs(
    validation_ids: np.ndarray,
    pair_count: int = 10,
    seed: int = 20260726,
) -> list[tuple[int, int]]:
    """Create fixed moving/fixed pairs; no pair contains the same patient twice."""
    if len(validation_ids) < 2:
        raise ValueError("At least two validation patients are required.")
    rng = np.random.default_rng(seed)
    moving_ids = rng.choice(validation_ids, size=pair_count, replace=True)
    fixed_ids = rng.choice(validation_ids, size=pair_count, replace=True)
    while np.any(moving_ids == fixed_ids):
        same_patient = moving_ids == fixed_ids
        fixed_ids[same_patient] = rng.choice(
            validation_ids, size=int(same_patient.sum()), replace=True
        )
    return list(zip(moving_ids.astype(int).tolist(), fixed_ids.astype(int).tolist()))


def masked_mse(target: torch.Tensor, prediction: torch.Tensor, mask=None, eps: float = 1e-6):
    error = (target - prediction).square()
    return error.mean() if mask is None else (error * mask).sum() / mask.sum().clamp_min(eps)


def masked_mae(target: torch.Tensor, prediction: torch.Tensor, mask=None, eps: float = 1e-6):
    error = (target - prediction).abs()
    return error.mean() if mask is None else (error * mask).sum() / mask.sum().clamp_min(eps)


def masked_ncc(target: torch.Tensor, prediction: torch.Tensor, mask=None, eps: float = 1e-6):
    """Global NCC over the lung-overlap voxels; larger is better."""
    if mask is None:
        mask = torch.ones_like(target)
    count = mask.sum().clamp_min(eps)
    target_mean = (target * mask).sum() / count
    prediction_mean = (prediction * mask).sum() / count
    target_centered = (target - target_mean) * mask
    prediction_centered = (prediction - prediction_mean) * mask
    return (target_centered * prediction_centered).sum() / torch.sqrt(
        target_centered.square().sum() * prediction_centered.square().sum()
    ).clamp_min(eps)


def evaluate_fixed_pairs(
    volumes: np.ndarray,
    lung_masks: np.ndarray | None,
    validation_pairs: list[tuple[int, int]],
    reconstruct_warped,
    device,
    output_dir: str | Path,
    epoch: int,
    representative_pair: int = 0,
) -> dict:
    """Evaluate fixed pairs and save aggregate CSV, pair CSV, and one PNG.

    `reconstruct_warped(moving, fixed)` must return the same tuple as the
    fine-tuning notebook.  It uses the current model parameters.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    representative = None

    with torch.no_grad():
        for pair_number, (moving_id, fixed_id) in enumerate(validation_pairs):
            moving = torch.from_numpy(volumes[moving_id:moving_id + 1]).unsqueeze(1).to(
                device=device, dtype=torch.float32
            )
            fixed = torch.from_numpy(volumes[fixed_id:fixed_id + 1]).unsqueeze(1).to(
                device=device, dtype=torch.float32
            )
            overlap = None
            if lung_masks is not None:
                moving_mask = torch.from_numpy(lung_masks[moving_id:moving_id + 1]).unsqueeze(1).to(
                    device=device, dtype=torch.float32
                )
                fixed_mask = torch.from_numpy(lung_masks[fixed_id:fixed_id + 1]).unsqueeze(1).to(
                    device=device, dtype=torch.float32
                )
                overlap = moving_mask * fixed_mask

            warped, flow, _, _, _ = reconstruct_warped(moving, fixed)
            metrics.append({
                "moving_id": moving_id,
                "fixed_id": fixed_id,
                "mse_before": masked_mse(fixed, moving, overlap).item(),
                "mse_after": masked_mse(fixed, warped, overlap).item(),
                "mae_before": masked_mae(fixed, moving, overlap).item(),
                "mae_after": masked_mae(fixed, warped, overlap).item(),
                "ncc_before": masked_ncc(fixed, moving, overlap).item(),
                "ncc_after": masked_ncc(fixed, warped, overlap).item(),
                "flow_abs_mean": flow.abs().mean().item(),
            })
            if pair_number == representative_pair:
                representative = (moving[0, 0].cpu(), fixed[0, 0].cpu(), warped[0, 0].cpu())

    numeric_columns = [
        "mse_before", "mse_after", "mae_before", "mae_after",
        "ncc_before", "ncc_after", "flow_abs_mean",
    ]
    summary = {column: float(np.mean([row[column] for row in metrics])) for column in numeric_columns}
    summary.update({"epoch": epoch, "pair_count": len(metrics)})

    summary_path = output_dir / "validation_metrics.csv"
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(summary)

    pairs_path = output_dir / f"validation_pairs_epoch_{epoch:05d}.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    moving, fixed, warped = representative
    slice_index = min(64, moving.shape[0] - 1)
    plt.figure(figsize=(12, 4))
    for panel, (title, image) in enumerate(
        [("Moving", moving[slice_index]), ("Fixed", fixed[slice_index]), ("Warped", warped[slice_index])], start=1
    ):
        plt.subplot(1, 3, panel)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    image_path = output_dir / f"validation_epoch_{epoch:05d}.png"
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"summary": summary, "summary_path": summary_path, "pairs_path": pairs_path, "image_path": image_path}
