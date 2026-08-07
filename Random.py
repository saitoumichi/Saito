"""Random smooth DVF equivalence test.

目的:
    A:
        Moving
        -> Full resolution smooth random DVF
        -> Full-resolution warped image

    B:
        Moving
        -> Wavelet Analysis
        -> Down Sampling
        -> Half-resolution smooth DVF
        -> Warp all 8 bands
        -> Synthesis
        -> Wavelet reconstructed warped image

    A と B がどの程度一致するか確認する。

確認:
    - MAE
    - MSE
    - RMSE
    - Max Absolute Error
    - Slice 32 / 48 / 64 / 80 / 96 の目視比較

注意:
    Full resolution: (128, 256, 256)
    Low resolution : (64, 128, 128)

    low DVF は full DVF の空間サイズを半分にした上で
    voxel単位の変位量も 0.5 倍する。
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
    "random_smooth_dvf_wavelet_check"
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

# 学習時と同じ粗い制御点
CONTROL_SHAPE = (
    8,
    16,
    16,
)

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# 乱数固定
RANDOM_SEED = 42

# フル解像度側での最大ランダム変位
SHIFT_RANGE = 4.0

# Gaussian smoothing
GAUSSIAN_SIGMA = 2.0

# まず何症例だけ試すか
# Noneなら全症例
MAX_CASES = 5


# =========================================================
# Gaussian smoothing
# =========================================================

def gaussian_smooth_3d(
    tensor,
    sigma=2.0
):
    from scipy.ndimage import gaussian_filter

    tensor_np = (
        tensor
        .detach()
        .cpu()
        .numpy()
    )

    smoothed_np = gaussian_filter(
        tensor_np,
        sigma=[
            0,
            0,
            sigma,
            sigma,
            sigma
        ]
    )

    return torch.tensor(
        smoothed_np,
        dtype=tensor.dtype,
        device=tensor.device
    )


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

def create_synthesis_filters(
    device
):

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
                band:
                band + 1
            ],
            filters[
                band:
                band + 1
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
# データ読み込み
# =========================================================

def load_single_volume(
    path
):

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


def load_test_pair_folders(
    root
):

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
            if (
                "fixed"
                in p.stem.lower()
            )
        ]

        if (
            moving_files
            and fixed_files
        ):

            moving_path = (
                moving_files[0]
            )

        elif len(files) == 2:

            moving_path = (
                files[0]
            )

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
# Random smooth DVF
# =========================================================

def create_random_smooth_dvf(
    batch_size,
    device,
    dtype
):

    # -----------------------------------------
    # 1. 粗い制御点でランダムDVF
    # -----------------------------------------

    control_dvf = (
        torch.rand(
            batch_size,
            3,
            CONTROL_SHAPE[0],
            CONTROL_SHAPE[1],
            CONTROL_SHAPE[2],
            device=device,
            dtype=dtype
        )
        * 2.0
        - 1.0
    )

    control_dvf = (
        control_dvf
        * SHIFT_RANGE
    )

    # -----------------------------------------
    # 2. Gaussian smoothing
    # -----------------------------------------

    control_dvf = (
        gaussian_smooth_3d(
            control_dvf,
            sigma=GAUSSIAN_SIGMA
        )
    )

    # -----------------------------------------
    # 3. Full resolutionへ補間
    # -----------------------------------------

    full_dvf = F.interpolate(
        control_dvf,
        size=FULL_SHAPE,
        mode="trilinear",
        align_corners=False
    )

    # -----------------------------------------
    # 4. Low resolutionへ補間
    # -----------------------------------------

    low_dvf = F.interpolate(
        full_dvf,
        size=LOW_SHAPE,
        mode="trilinear",
        align_corners=False
    )

    # 重要:
    # Full 1 voxel
    # =
    # Low 0.5 voxel
    low_dvf = (
        low_dvf
        * 0.5
    )

    return (
        control_dvf,
        full_dvf,
        low_dvf
    )


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    reference,
    prediction
):

    difference = (
        prediction
        - reference
    )

    mae = (
        difference
        .abs()
        .mean()
        .item()
    )

    mse = (
        difference
        .square()
        .mean()
        .item()
    )

    rmse = float(
        np.sqrt(mse)
    )

    max_abs = (
        difference
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
# DVF statistics
# =========================================================

def dvf_statistics(
    dvf
):

    return {
        "mean": (
            dvf.mean().item()
        ),
        "std": (
            dvf.std().item()
        ),
        "min": (
            dvf.min().item()
        ),
        "max": (
            dvf.max().item()
        ),
        "max_abs": (
            dvf.abs()
            .max()
            .item()
        ),
    }


# =========================================================
# 可視化
# =========================================================

def save_figures(
    case_id,
    case_name,
    original,
    warped_full,
    warped_wavelet
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

    for slice_idx in (
        SLICE_INDICES
    ):

        if (
            slice_idx
            >= original.shape[2]
        ):

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
            warped_full[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        wavelet_slice = (
            warped_wavelet[
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

        vmin = (
            torch.quantile(
                display_values,
                0.01
            )
            .item()
        )

        vmax = (
            torch.quantile(
                display_values,
                0.99
            )
            .item()
        )

        difference_limit = max(
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

        # Full Warp
        axes[1].imshow(
            full_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[1].set_title(
            "Full-resolution smooth DVF"
        )

        # Wavelet Warp
        axes[2].imshow(
            wavelet_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[2].set_title(
            "Wavelet low-res smooth DVF"
        )

        # Difference
        image = axes[3].imshow(
            difference,
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit
        )

        axes[3].set_title(
            "Wavelet warp − Full warp"
        )

        for axis in axes:
            axis.axis(
                "off"
            )

        figure.colorbar(
            image,
            ax=axes[3],
            fraction=0.046,
            pad=0.04
        )

        figure.suptitle(
            (
                f"{case_name} "
                f"Random Smooth DVF Check "
                f"— Slice {slice_idx}"
            ),
            fontsize=16
        )

        save_path = (
            case_dir
            /
            f"slice_{slice_idx:03d}_random_dvf.png"
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
# Overview
# =========================================================

def save_overview(
    case_id,
    case_name,
    original,
    warped_full,
    warped_wavelet
):

    valid_slices = [
        s
        for s in SLICE_INDICES
        if s < original.shape[2]
    ]

    case_dir = (
        OUTPUT_DIR
        /
        f"case_{case_id:03d}_{case_name}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure, axes = plt.subplots(
        len(valid_slices),
        4,
        figsize=(
            16,
            4 * len(valid_slices)
        ),
        constrained_layout=True
    )

    for row_idx, slice_idx in enumerate(
        valid_slices
    ):

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
            warped_full[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        wavelet_slice = (
            warped_wavelet[
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

        difference_limit = max(
            torch.quantile(
                difference.abs(),
                0.99
            ).item(),
            1e-8
        )

        images = [
            original_slice,
            full_slice,
            wavelet_slice,
            difference
        ]

        for col_idx, image in enumerate(
            images
        ):

            axis = axes[
                row_idx,
                col_idx
            ]

            if col_idx < 3:

                axis.imshow(
                    image,
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax
                )

            else:

                axis.imshow(
                    image,
                    cmap="coolwarm",
                    vmin=-difference_limit,
                    vmax=difference_limit
                )

            axis.set_xticks([])
            axis.set_yticks([])

        axes[
            row_idx,
            0
        ].set_ylabel(
            f"Slice {slice_idx}"
        )

    axes[
        0,
        0
    ].set_title(
        "Original"
    )

    axes[
        0,
        1
    ].set_title(
        "Full DVF"
    )

    axes[
        0,
        2
    ].set_title(
        "Wavelet DVF"
    )

    axes[
        0,
        3
    ].set_title(
        "Difference"
    )

    figure.suptitle(
        (
            f"{case_name} "
            f"Random Smooth DVF Equivalence"
        ),
        fontsize=18
    )

    save_path = (
        case_dir
        /
        "random_dvf_overview.png"
    )

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
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
        "Random seed:",
        RANDOM_SEED
    )

    print(
        "Shift range:",
        SHIFT_RANGE
    )

    print(
        "Gaussian sigma:",
        GAUSSIAN_SIGMA
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_cases = (
        load_test_pair_folders(
            TEST_PAIR_ROOT
        )
    )

    if MAX_CASES is not None:

        test_cases = (
            test_cases[
                :MAX_CASES
            ]
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
        ) in enumerate(
            test_cases
        ):

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

            # =============================================
            # DVF生成
            # =============================================

            (
                control_dvf,
                full_dvf,
                low_dvf
            ) = create_random_smooth_dvf(
                batch_size=
                moving.shape[0],
                device=
                DEVICE,
                dtype=
                moving.dtype
            )

            print(
                "control_dvf:",
                control_dvf.shape
            )

            print(
                "full_dvf:",
                full_dvf.shape
            )

            print(
                "low_dvf:",
                low_dvf.shape
            )

            full_stats = (
                dvf_statistics(
                    full_dvf
                )
            )

            low_stats = (
                dvf_statistics(
                    low_dvf
                )
            )

            print(
                "\nFull DVF stats:",
                full_stats
            )

            print(
                "Low DVF stats:",
                low_stats
            )

            # =============================================
            # A:
            # Full resolutionでWarp
            # =============================================

            warped_full = (
                full_transformer(
                    moving,
                    full_dvf
                )
            )

            # =============================================
            # B:
            # Wavelet -> Low DVF -> Synthesis
            # =============================================

            analyzed = (
                analysis(
                    moving
                )
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

            warped_bands = []

            for channel in range(8):

                warped_band = (
                    low_transformer(
                        moving_bands[
                            :,
                            channel:
                            channel + 1
                        ],
                        low_dvf
                    )
                )

                warped_bands.append(
                    warped_band
                )

            warped_bands = (
                torch.cat(
                    warped_bands,
                    dim=1
                )
            )

            warped_wavelet = (
                synthesize(
                    warped_bands,
                    synthesis_filters
                )
            )

            # =============================================
            # Metrics
            # =============================================

            result = (
                calculate_metrics(
                    warped_full,
                    warped_wavelet
                )
            )

            print(
                "\nRandom Smooth DVF Comparison"
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

            # =============================================
            # Save figures
            # =============================================

            save_figures(
                case_id,
                case_name,
                moving,
                warped_full,
                warped_wavelet
            )

            save_overview(
                case_id,
                case_name,
                moving,
                warped_full,
                warped_wavelet
            )

            # =============================================
            # CSV
            # =============================================

            row = {
                "case_id":
                    case_id,

                "case_name":
                    case_name,

                "shift_range":
                    SHIFT_RANGE,

                "gaussian_sigma":
                    GAUSSIAN_SIGMA,

                "full_dvf_mean":
                    full_stats["mean"],

                "full_dvf_std":
                    full_stats["std"],

                "full_dvf_max_abs":
                    full_stats["max_abs"],

                "low_dvf_mean":
                    low_stats["mean"],

                "low_dvf_std":
                    low_stats["std"],

                "low_dvf_max_abs":
                    low_stats["max_abs"],

                **result
            }

            summary_rows.append(
                row
            )

            case_dir = (
                OUTPUT_DIR
                /
                f"case_{case_id:03d}_{case_name}"
            )

            case_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            csv_path = (
                case_dir
                /
                "random_smooth_dvf_metrics.csv"
            )

            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=
                    row.keys()
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

        writer = (
            csv.DictWriter(
                file,
                fieldnames=
                summary_rows[0].keys()
            )
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
            for row
            in summary_rows
        ]

        print(
            f"Mean "
            f"{key.upper():7s} "
            f"= "
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