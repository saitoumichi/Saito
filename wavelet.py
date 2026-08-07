"""Constant shift test for Wavelet deformation.

目的:
    フル解像度で +2 voxel 移動した画像と、
    Wavelet後の低解像度で +1 voxel 移動して再合成した画像が
    どの程度一致するか確認する。

比較:
    A: Original Moving -> Full-resolution shift +2 voxel
    B: Original Moving -> Wavelet -> Low-resolution shift +1 voxel
                         -> Synthesis

出力:
    constant_shift_wavelet_check/
        case_000_pair1/
            slice_032_constant_shift.png
            slice_048_constant_shift.png
            ...
            constant_shift_metrics.csv
        summary.csv
"""

import csv
import os
from pathlib import Path

os.environ["VXM_BACKEND"] = "pytorch"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import voxelmorph as vxm


# =========================================================
# 設定
# =========================================================

TEST_PAIR_ROOT = Path("Data/TestData")

OUTPUT_DIR = Path(
    "constant_shift_wavelet_check"
)

SLICE_INDICES = [
    32,
    48,
    64,
    80,
    96,
]

FULL_SHAPE = (
    128,
    256,
    256,
)

LOW_SHAPE = (
    64,
    128,
    128,
)

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# -----------------------------------------
# 一定変位
# -----------------------------------------

# FULL側では2 voxel
FULL_SHIFT = 2.0

# LOW側では1 voxel
LOW_SHIFT = 1.0

# 0=z, 1=y, 2=x
SHIFT_AXIS = 2


# =========================================================
# Haar Analysis
# =========================================================

class Haar3DAnalysisOnly(nn.Module):

    def __init__(self):
        super().__init__()

        low = (
            torch.tensor(
                [1.0, 1.0]
            )
            / np.sqrt(2.0)
        )

        high = (
            torch.tensor(
                [1.0, -1.0]
            )
            / np.sqrt(2.0)
        )

        filters = [
            z[:, None, None]
            * y[None, :, None]
            * x[None, None, :]
            for z in (low, high)
            for y in (low, high)
            for x in (low, high)
        ]

        self.register_buffer(
            "weight",
            torch.stack(filters)
            .unsqueeze(1)
            .float()
        )

    def forward(self, image):

        return F.conv3d(
            F.pad(
                image,
                (
                    0, 1,
                    0, 1,
                    0, 1
                )
            ),
            self.weight,
            stride=1
        )


# =========================================================
# Synthesis
# =========================================================

def create_synthesis_filters(device):

    low = (
        torch.tensor(
            [1.0, 1.0],
            device=device
        )
        / np.sqrt(2.0)
    )

    high = (
        torch.tensor(
            [1.0, -1.0],
            device=device
        )
        / np.sqrt(2.0)
    )

    filters = [
        z[:, None, None]
        * y[None, :, None]
        * x[None, None, :]
        for z in (low, high)
        for y in (low, high)
        for x in (low, high)
    ]

    return torch.flip(
        torch.stack(filters),
        dims=[1, 2, 3]
    ).unsqueeze(1)


def synthesize(
    wavelet_bands,
    filters
):

    up = torch.zeros(
        wavelet_bands.shape[0],
        wavelet_bands.shape[1],
        wavelet_bands.shape[2] * 2,
        wavelet_bands.shape[3] * 2,
        wavelet_bands.shape[4] * 2,
        device=wavelet_bands.device,
        dtype=wavelet_bands.dtype
    )

    up[
        :,
        :,
        ::2,
        ::2,
        ::2
    ] = wavelet_bands

    reconstructed = []

    for band in range(8):

        filtered = F.conv3d(
            up[
                :,
                band:band + 1
            ],
            filters[
                band:band + 1
            ],
            padding=1
        )

        filtered = filtered[
            :,
            :,
            :up.shape[2],
            :up.shape[3],
            :up.shape[4]
        ]

        reconstructed.append(
            filtered
        )

    return (
        torch.cat(
            reconstructed,
            dim=1
        )
        .sum(
            dim=1,
            keepdim=True
        )
    )


# =========================================================
# 読み込み
# =========================================================

def load_single_volume(path):

    if path.suffix.lower() == ".npy":

        volume = np.load(path)

    elif path.suffix.lower() == ".npz":

        archive = np.load(path)

        if len(
            archive.files
        ) != 1:

            raise ValueError(
                f"{path} は複数配列を含みます"
            )

        volume = archive[
            archive.files[0]
        ]

    else:

        raise ValueError(
            f"未対応形式: {path}"
        )

    volume = np.squeeze(
        volume
    )

    if tuple(
        volume.shape
    ) != FULL_SHAPE:

        raise ValueError(
            f"{path.name}: "
            f"{volume.shape} != {FULL_SHAPE}"
        )

    return volume.astype(
        np.float32
    )


def load_test_pair_folders(root):

    if not root.is_dir():

        raise FileNotFoundError(
            f"TestDataがありません: "
            f"{root.resolve()}"
        )

    cases = []

    for pair_dir in sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
    ):

        files = sorted(
            list(
                pair_dir.glob("*.npy")
            )
            +
            list(
                pair_dir.glob("*.npz")
            )
        )

        moving_files = [
            p
            for p in files
            if (
                "moving"
                in p.stem.lower()
                or
                "registered"
                in p.stem.lower()
            )
        ]

        fixed_files = [
            p
            for p in files
            if "fixed"
            in p.stem.lower()
        ]

        if (
            moving_files
            and fixed_files
        ):

            moving_path = moving_files[0]

        elif len(files) == 2:

            moving_path = files[0]

            print(
                f"[注意] "
                f"{pair_dir.name}: "
                f"名前順でMovingを使用 "
                f"{moving_path.name}"
            )

        else:

            print(
                f"[スキップ] "
                f"{pair_dir.name}"
            )

            continue

        cases.append(
            (
                pair_dir.name,
                load_single_volume(
                    moving_path
                )
            )
        )

    if not cases:

        raise RuntimeError(
            "評価できる症例がありません"
        )

    return cases


# =========================================================
# 一定DVF作成
# =========================================================

def make_constant_flow(
    batch_size,
    shape,
    shift,
    axis,
    device,
    dtype
):

    flow = torch.zeros(
        batch_size,
        3,
        shape[0],
        shape[1],
        shape[2],
        device=device,
        dtype=dtype
    )

    flow[
        :,
        axis,
        :,
        :,
        :
    ] = shift

    return flow


# =========================================================
# metrics
# =========================================================

def calculate_metrics(
    reference,
    prediction
):

    diff = (
        prediction
        - reference
    )

    mae = (
        diff
        .abs()
        .mean()
        .item()
    )

    mse = (
        diff
        .square()
        .mean()
        .item()
    )

    rmse = float(
        np.sqrt(mse)
    )

    max_abs = (
        diff
        .abs()
        .max()
        .item()
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs": max_abs,
    }


# =========================================================
# 可視化
# =========================================================

def save_figures(
    case_id,
    case_name,
    original,
    shifted_full,
    shifted_wavelet
):

    case_dir = (
        OUTPUT_DIR
        /
        f"case_{case_id:03d}_{case_name}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for slice_idx in SLICE_INDICES:

        if slice_idx >= original.shape[2]:
            continue

        original_slice = (
            original[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        full_slice = (
            shifted_full[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        wavelet_slice = (
            shifted_wavelet[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        difference = (
            wavelet_slice
            - full_slice
        )

        display_values = torch.cat(
            [
                original_slice.flatten(),
                full_slice.flatten(),
                wavelet_slice.flatten()
            ]
        )

        vmin = torch.quantile(
            display_values,
            0.01
        ).item()

        vmax = torch.quantile(
            display_values,
            0.99
        ).item()

        diff_limit = max(
            torch.quantile(
                difference.abs(),
                0.99
            ).item(),
            1e-8
        )

        figure, axes = plt.subplots(
            1,
            4,
            figsize=(
                18,
                5
            ),
            constrained_layout=True
        )

        # Original
        axes[0].imshow(
            original_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[0].set_title(
            "Original Moving"
        )

        # Full shift
        axes[1].imshow(
            full_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[1].set_title(
            f"Full resolution shift +{FULL_SHIFT}"
        )

        # Wavelet shift
        axes[2].imshow(
            wavelet_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[2].set_title(
            f"Wavelet low-res shift +{LOW_SHIFT}"
        )

        # Difference
        im = axes[3].imshow(
            difference,
            cmap="coolwarm",
            vmin=-diff_limit,
            vmax=diff_limit
        )

        axes[3].set_title(
            "Wavelet shift − Full shift"
        )

        for ax in axes:
            ax.axis("off")

        figure.colorbar(
            im,
            ax=axes[3],
            fraction=0.046,
            pad=0.04
        )

        figure.suptitle(
            (
                f"{case_name} "
                f"Constant Shift Check "
                f"— Slice {slice_idx}"
            ),
            fontsize=16
        )

        save_path = (
            case_dir
            /
            f"slice_{slice_idx:03d}_constant_shift.png"
        )

        figure.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(
            figure
        )

        print(
            "保存:",
            save_path
        )


# =========================================================
# main
# =========================================================

def main():

    print(
        "Device:",
        DEVICE
    )

    print(
        "FULL shift:",
        FULL_SHIFT
    )

    print(
        "LOW shift:",
        LOW_SHIFT
    )

    print(
        "SHIFT_AXIS:",
        SHIFT_AXIS
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cases = load_test_pair_folders(
        TEST_PAIR_ROOT
    )

    analysis = (
        Haar3DAnalysisOnly()
        .to(
            DEVICE
        )
    )

    synthesis_filters = (
        create_synthesis_filters(
            DEVICE
        )
    )

    full_transformer = (
        vxm.layers
        .SpatialTransformer(
            FULL_SHAPE
        )
        .to(
            DEVICE
        )
    )

    low_transformer = (
        vxm.layers
        .SpatialTransformer(
            LOW_SHAPE
        )
        .to(
            DEVICE
        )
    )

    summary_rows = []

    with torch.no_grad():

        for case_id, (
            case_name,
            moving_volume
        ) in enumerate(cases):

            print(
                "\n===================================="
            )

            print(
                f"Case {case_id}: "
                f"{case_name}"
            )

            print(
                "===================================="
            )

            moving = (
                torch.from_numpy(
                    moving_volume[
                        None,
                        None
                    ]
                )
                .to(
                    DEVICE
                )
            )

            # =================================================
            # A:
            # Full resolutionで+2 voxel移動
            # =================================================

            full_flow = (
                make_constant_flow(
                    batch_size=moving.shape[0],
                    shape=FULL_SHAPE,
                    shift=FULL_SHIFT,
                    axis=SHIFT_AXIS,
                    device=DEVICE,
                    dtype=moving.dtype
                )
            )

            shifted_full = (
                full_transformer(
                    moving,
                    full_flow
                )
            )

            # =================================================
            # B:
            # Wavelet
            # -> low resolutionで+1 voxel
            # -> synthesis
            # =================================================

            analyzed = analysis(
                moving
            )

            moving_bands = (
                analyzed[
                    :,
                    :,
                    ::2,
                    ::2,
                    ::2
                ]
            )

            low_flow = (
                make_constant_flow(
                    batch_size=moving.shape[0],
                    shape=LOW_SHAPE,
                    shift=LOW_SHIFT,
                    axis=SHIFT_AXIS,
                    device=DEVICE,
                    dtype=moving.dtype
                )
            )

            warped_bands = []

            for channel in range(8):

                warped_band = (
                    low_transformer(
                        moving_bands[
                            :,
                            channel:
                            channel + 1
                        ],
                        low_flow
                    )
                )

                warped_bands.append(
                    warped_band
                )

            warped_bands = torch.cat(
                warped_bands,
                dim=1
            )

            shifted_wavelet = (
                synthesize(
                    warped_bands,
                    synthesis_filters
                )
            )

            # =================================================
            # Metrics
            # =================================================

            result = (
                calculate_metrics(
                    shifted_full,
                    shifted_wavelet
                )
            )

            print(
                "\nConstant Shift Comparison"
            )

            print(
                f"MAE     = "
                f"{result['mae']:.10e}"
            )

            print(
                f"MSE     = "
                f"{result['mse']:.10e}"
            )

            print(
                f"RMSE    = "
                f"{result['rmse']:.10e}"
            )

            print(
                f"Max Abs = "
                f"{result['max_abs']:.10e}"
            )

            save_figures(
                case_id,
                case_name,
                moving,
                shifted_full,
                shifted_wavelet
            )

            row = {
                "case_id": case_id,
                "case_name": case_name,
                "full_shift": FULL_SHIFT,
                "low_shift": LOW_SHIFT,
                "shift_axis": SHIFT_AXIS,
                **result,
            }

            summary_rows.append(
                row
            )

            # 1症例ごとのCSV
            case_dir = (
                OUTPUT_DIR
                /
                f"case_{case_id:03d}_{case_name}"
            )

            csv_path = (
                case_dir
                /
                "constant_shift_metrics.csv"
            )

            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=row.keys()
                )

                writer.writeheader()

                writer.writerow(
                    row
                )

    # =====================================================
    # Summary
    # =====================================================

    summary_path = (
        OUTPUT_DIR
        /
        "summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
            summary_rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    print(
        "\n===================================="
    )

    print(
        "All Cases Summary"
    )

    print(
        "===================================="
    )

    for key in (
        "mae",
        "mse",
        "rmse",
        "max_abs"
    ):

        values = [
            row[key]
            for row in summary_rows
        ]

        print(
            f"Mean {key.upper():7s} = "
            f"{np.mean(values):.10e}"
        )

    print(
        "\n保存先:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()