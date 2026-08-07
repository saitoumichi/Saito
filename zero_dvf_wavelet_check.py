"""Zero DVF Wavelet reconstruction check.

目的:
    Moving画像にゼロ変位場を与えて、

        Moving
        -> Wavelet Analysis
        -> Down Sampling
        -> Zero DVF Warp
        -> Up Sampling
        -> Synthesis

    を通した画像が、元のMovingとどの程度一致するか確認する。

確認内容:
    - MAE
    - MSE
    - RMSE
    - Max Absolute Error
    - Slice 32 / 48 / 64 / 80 / 96 の目視比較

出力:
    zero_dvf_wavelet_check/
        case_000_pair1/
            slice_032_zero_dvf.png
            slice_048_zero_dvf.png
            ...
            zero_dvf_metrics.csv
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
# ユーザー設定
# =========================================================

TEST_PAIR_ROOT = Path("Data/TestData")

OUTPUT_DIR = Path(
    "zero_dvf_wavelet_check"
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


# =========================================================
# Haar 3D Analysis
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
            for z in (
                low,
                high
            )
            for y in (
                low,
                high
            )
            for x in (
                low,
                high
            )
        ]

        weight = (
            torch.stack(
                filters
            )
            .unsqueeze(1)
            .float()
        )

        self.register_buffer(
            "weight",
            weight
        )

    def forward(
        self,
        image
    ):

        # 元の比較コードと同じAnalysis処理
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
# Synthesis Filter
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
        for z in (
            low,
            high
        )
        for y in (
            low,
            high
        )
        for x in (
            low,
            high
        )
    ]

    filters = torch.stack(
        filters
    )

    filters = torch.flip(
        filters,
        dims=[
            1,
            2,
            3
        ]
    )

    filters = filters.unsqueeze(
        1
    )

    return filters


# =========================================================
# Synthesis
# =========================================================

def synthesize(
    wavelet_bands,
    filters
):

    batch_size = (
        wavelet_bands.shape[0]
    )

    channels = (
        wavelet_bands.shape[1]
    )

    depth = (
        wavelet_bands.shape[2]
        * 2
    )

    height = (
        wavelet_bands.shape[3]
        * 2
    )

    width = (
        wavelet_bands.shape[4]
        * 2
    )

    up = torch.zeros(
        batch_size,
        channels,
        depth,
        height,
        width,
        device=wavelet_bands.device,
        dtype=wavelet_bands.dtype
    )

    # Up Sampling:
    # 偶数位置にWavelet係数を配置
    up[
        :,
        :,
        ::2,
        ::2,
        ::2
    ] = wavelet_bands

    reconstructed_bands = []

    for band in range(
        8
    ):

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

        # 元サイズにcrop
        filtered = filtered[
            :,
            :,
            :up.shape[2],
            :up.shape[3],
            :up.shape[4]
        ]

        reconstructed_bands.append(
            filtered
        )

    reconstructed = (
        torch.cat(
            reconstructed_bands,
            dim=1
        )
        .sum(
            dim=1,
            keepdim=True
        )
    )

    return reconstructed


# =========================================================
# 1症例の読み込み
# =========================================================

def load_single_volume(
    path
):

    suffix = (
        path.suffix.lower()
    )

    if suffix == ".npy":

        volume = np.load(
            path
        )

    elif suffix == ".npz":

        archive = np.load(
            path
        )

        if len(
            archive.files
        ) != 1:

            raise ValueError(
                f"{path} は複数配列を含みます。"
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
            f"shape={volume.shape}, "
            f"expected={FULL_SHAPE}"
        )

    return volume.astype(
        np.float32
    )


# =========================================================
# TestData pairフォルダ読み込み
# =========================================================

def load_test_pair_folders(
    root
):

    if not root.is_dir():

        raise FileNotFoundError(
            f"TestDataフォルダが"
            f"見つかりません: "
            f"{root.resolve()}"
        )

    cases = []

    pair_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
    )

    for pair_dir in pair_dirs:

        files = sorted(
            list(
                pair_dir.glob(
                    "*.npy"
                )
            )
            +
            list(
                pair_dir.glob(
                    "*.npz"
                )
            )
        )

        moving_files = [
            path
            for path in files
            if (
                "moving"
                in path.stem.lower()
                or
                "registered"
                in path.stem.lower()
            )
        ]

        fixed_files = [
            path
            for path in files
            if (
                "fixed"
                in path.stem.lower()
            )
        ]

        if (
            moving_files
            and fixed_files
        ):

            moving_path = (
                moving_files[0]
            )

            fixed_path = (
                fixed_files[0]
            )

        elif len(
            files
        ) == 2:

            moving_path = files[0]
            fixed_path = files[1]

            print(
                f"[注意] "
                f"{pair_dir.name}: "
                f"Moving/Fixed名なし。"
                f"名前順で使用:"
            )

            print(
                "  Moving:",
                moving_path.name
            )

            print(
                "  Fixed :",
                fixed_path.name
            )

        else:

            print(
                f"[スキップ] "
                f"{pair_dir.name}: "
                f"Moving/Fixedを"
                f"特定できません"
            )

            continue

        moving = load_single_volume(
            moving_path
        )

        fixed = load_single_volume(
            fixed_path
        )

        cases.append(
            (
                pair_dir.name,
                moving,
                fixed
            )
        )

    if not cases:

        raise RuntimeError(
            "評価可能なTest pairが"
            "見つかりませんでした。"
        )

    return cases


# =========================================================
# 数値評価
# =========================================================

def calculate_metrics(
    original,
    reconstructed
):

    difference = (
        reconstructed
        - original
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
        np.sqrt(
            mse
        )
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
# Slice画像保存
# =========================================================

def save_slice_comparisons(
    case_id,
    case_name,
    original,
    reconstructed
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

        if slice_idx >= (
            original.shape[2]
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

        reconstructed_slice = (
            reconstructed[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        difference_slice = (
            reconstructed_slice
            - original_slice
        )

        display_values = torch.cat(
            [
                original_slice.flatten(),
                reconstructed_slice.flatten()
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
                difference_slice.abs(),
                0.99
            ).item(),
            1e-8
        )

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(
                15,
                5
            ),
            constrained_layout=True
        )

        # Original Moving
        axes[0].imshow(
            original_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[0].set_title(
            "Original Moving"
        )

        axes[0].axis(
            "off"
        )

        # Reconstructed
        axes[1].imshow(
            reconstructed_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        axes[1].set_title(
            "Wavelet → Zero DVF → Synthesis"
        )

        axes[1].axis(
            "off"
        )

        # Difference
        im = axes[2].imshow(
            difference_slice,
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit
        )

        axes[2].set_title(
            "Reconstructed − Original"
        )

        axes[2].axis(
            "off"
        )

        figure.colorbar(
            im,
            ax=axes[2],
            fraction=0.046,
            pad=0.04
        )

        figure.suptitle(
            f"{case_name} "
            f"— Slice {slice_idx}",
            fontsize=16
        )

        save_path = (
            case_dir
            /
            f"slice_{slice_idx:03d}_zero_dvf.png"
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
# 5スライスまとめ画像
# =========================================================

def save_overview(
    case_id,
    case_name,
    original,
    reconstructed
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
        3,
        figsize=(
            12,
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

        reconstructed_slice = (
            reconstructed[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        difference_slice = (
            reconstructed_slice
            - original_slice
        )

        display_values = torch.cat(
            [
                original_slice.flatten(),
                reconstructed_slice.flatten()
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
                difference_slice.abs(),
                0.99
            ).item(),
            1e-8
        )

        # Original
        axes[
            row_idx,
            0
        ].imshow(
            original_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        # Reconstructed
        axes[
            row_idx,
            1
        ].imshow(
            reconstructed_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        # Difference
        axes[
            row_idx,
            2
        ].imshow(
            difference_slice,
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit
        )

        for col_idx in range(
            3
        ):

            axes[
                row_idx,
                col_idx
            ].set_xticks([])

            axes[
                row_idx,
                col_idx
            ].set_yticks([])

        axes[
            row_idx,
            0
        ].set_ylabel(
            f"Slice "
            f"{slice_idx}",
            fontsize=12
        )

    axes[
        0,
        0
    ].set_title(
        "Original Moving"
    )

    axes[
        0,
        1
    ].set_title(
        "Wavelet → Zero DVF → Synthesis"
    )

    axes[
        0,
        2
    ].set_title(
        "Difference"
    )

    figure.suptitle(
        f"{case_name} "
        f"Zero DVF Wavelet Check",
        fontsize=18
    )

    save_path = (
        case_dir
        /
        "zero_dvf_overview.png"
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
# 1症例処理
# =========================================================

def run_zero_dvf_check(
    case_id,
    case_name,
    moving,
    analysis,
    low_transformer,
    synthesis_filters
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

    # -----------------------------------------
    # Analysis
    # -----------------------------------------

    analyzed = analysis(
        moving
    )

    print(
        "Analysis output:",
        analyzed.shape
    )

    # -----------------------------------------
    # Down Sampling
    # -----------------------------------------

    moving_bands = (
        analyzed[
            :,
            :,
            ::2,
            ::2,
            ::2
        ]
    )

    print(
        "Down sampled bands:",
        moving_bands.shape
    )

    # -----------------------------------------
    # Zero DVF
    # -----------------------------------------

    zero_flow = torch.zeros(
        moving.shape[0],
        3,
        moving_bands.shape[2],
        moving_bands.shape[3],
        moving_bands.shape[4],
        device=moving.device,
        dtype=moving.dtype
    )

    print(
        "Zero flow:",
        zero_flow.shape
    )

    print(
        "Zero flow max abs:",
        zero_flow.abs().max().item()
    )

    # -----------------------------------------
    # Warp each Wavelet band
    # -----------------------------------------

    warped_bands = []

    for channel in range(
        8
    ):

        band = (
            moving_bands[
                :,
                channel:
                channel + 1
            ]
        )

        warped = (
            low_transformer(
                band,
                zero_flow
            )
        )

        warped_bands.append(
            warped
        )

    warped_bands = torch.cat(
        warped_bands,
        dim=1
    )

    print(
        "Warped bands:",
        warped_bands.shape
    )

    # -----------------------------------------
    # Synthesis
    # -----------------------------------------

    reconstructed = synthesize(
        warped_bands,
        synthesis_filters
    )

    print(
        "Reconstructed:",
        reconstructed.shape
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    result = calculate_metrics(
        moving,
        reconstructed
    )

    print(
        "\nZero DVF Reconstruction Metrics"
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

    # -----------------------------------------
    # Visual comparison
    # -----------------------------------------

    save_slice_comparisons(
        case_id,
        case_name,
        moving,
        reconstructed
    )

    save_overview(
        case_id,
        case_name,
        moving,
        reconstructed
    )

    return (
        reconstructed,
        result
    )


# =========================================================
# main
# =========================================================

def main():

    print(
        "Device:",
        DEVICE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # Test pairs
    # -----------------------------------------

    test_cases = (
        load_test_pair_folders(
            TEST_PAIR_ROOT
        )
    )

    print(
        "Number of test pairs:",
        len(
            test_cases
        )
    )

    # -----------------------------------------
    # Wavelet modules
    # -----------------------------------------

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

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    with torch.no_grad():

        for case_id, (
            case_name,
            moving_volume,
            fixed_volume
        ) in enumerate(
            test_cases
        ):

            # Fixedは今回は使わない
            # MovingのみZero DVF再構成する

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

            _, result = (
                run_zero_dvf_check(
                    case_id,
                    case_name,
                    moving,
                    analysis,
                    low_transformer,
                    synthesis_filters
                )
            )

            row = {
                "case_id": case_id,
                "case_name": case_name,
                **result
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

            case_csv = (
                case_dir
                /
                "zero_dvf_metrics.csv"
            )

            with case_csv.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = (
                    csv.DictWriter(
                        file,
                        fieldnames=row.keys()
                    )
                )

                writer.writeheader()

                writer.writerow(
                    row
                )

    # -----------------------------------------
    # 全症例Summary
    # -----------------------------------------

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

    # -----------------------------------------
    # 全症例平均
    # -----------------------------------------

    mean_mae = np.mean(
        [
            row["mae"]
            for row
            in summary_rows
        ]
    )

    mean_mse = np.mean(
        [
            row["mse"]
            for row
            in summary_rows
        ]
    )

    mean_rmse = np.mean(
        [
            row["rmse"]
            for row
            in summary_rows
        ]
    )

    mean_max_abs = np.mean(
        [
            row["max_abs"]
            for row
            in summary_rows
        ]
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

    print(
        f"Mean MAE     = "
        f"{mean_mae:.10e}"
    )

    print(
        f"Mean MSE     = "
        f"{mean_mse:.10e}"
    )

    print(
        f"Mean RMSE    = "
        f"{mean_rmse:.10e}"
    )

    print(
        f"Mean Max Abs = "
        f"{mean_max_abs:.10e}"
    )

    print(
        "\n結果保存先:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()