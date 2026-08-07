"""128モデルと256+Waveletモデルを、同一の実テストペアで比較する評価スクリプト。

実行前に TEST_DATA_PATH、CHECKPOINT_128_PATH、CHECKPOINT_WAVELET_PATH、TEST_PAIRS を設定する。
TEST_PAIRS は (moving_index, fixed_index) のリストで、実際に比較したい対応ペアを指定する。
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

# pair別フォルダを使う場合は None
TEST_DATA_PATH = None

TEST_PAIR_ROOT = Path("Data/TestData")

CHECKPOINT_128_PATH = Path(
    "yamatoCode/a.pth"
)

CHECKPOINT_WAVELET_PATH = Path(
    "curriculum_checkpoints_dvf_scale_corrected/pretrain_epoch_80000.pth"
)

# TEST_DATA_PATH を使う場合のみ使用
TEST_PAIRS = [
    (0, 1),
    (2, 3),
    (4, 5),
]

OUTPUT_DIR = Path(
    "comparison_128_vs_256_wavelet"
)

# CTで目視確認するスライス
SLICE_INDICES = [
    32,
    48,
    64,
    80,
    96,
]

# =========================================================


DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

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

NB_FEATURES = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16],
]


# =========================================================
# Haar 3D Analysis
# =========================================================

class Haar3DAnalysisOnly(nn.Module):

    def __init__(self):
        super().__init__()

        low = (
            torch.tensor([1.0, 1.0])
            / np.sqrt(2.0)
        )

        high = (
            torch.tensor([1.0, -1.0])
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
                (0, 1, 0, 1, 0, 1)
            ),
            self.weight,
            stride=1,
        )


# =========================================================
# Synthesis filter
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


# =========================================================
# Wavelet synthesis
# =========================================================

def synthesize(
    wavelet_bands,
    filters,
):

    up = torch.zeros(
        wavelet_bands.shape[0],
        wavelet_bands.shape[1],
        wavelet_bands.shape[2] * 2,
        wavelet_bands.shape[3] * 2,
        wavelet_bands.shape[4] * 2,
        device=wavelet_bands.device,
        dtype=wavelet_bands.dtype,
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
            padding=1,
        )

        filtered = filtered[
            :,
            :,
            :up.shape[2],
            :up.shape[3],
            :up.shape[4],
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
# checkpoint読み込み
# =========================================================

def state_dict(path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    return checkpoint.get(
        "model_state_dict",
        checkpoint
    )


# =========================================================
# 評価指標
# =========================================================

def ncc(
    reference,
    prediction,
    eps=1e-8,
):

    reference = (
        reference
        - reference.mean()
    )

    prediction = (
        prediction
        - prediction.mean()
    )

    return (
        (
            reference
            * prediction
        ).sum()
        /
        torch.sqrt(
            reference.square().sum()
            * prediction.square().sum()
            + eps
        )
    ).item()


def gradient_mae(
    reference,
    prediction,
):

    errors = []

    for axis in (
        2,
        3,
        4,
    ):

        reference_gradient = torch.diff(
            reference,
            dim=axis
        )

        prediction_gradient = torch.diff(
            prediction,
            dim=axis
        )

        errors.append(
            (
                reference_gradient
                - prediction_gradient
            )
            .abs()
            .mean()
        )

    return (
        torch.stack(errors)
        .mean()
        .item()
    )


def metrics(
    reference,
    prediction,
):

    mse = torch.mean(
        (
            reference
            - prediction
        ) ** 2
    ).item()

    return {
        "mse": mse,
        "rmse": float(
            np.sqrt(mse)
        ),
        "ncc": ncc(
            reference,
            prediction
        ),
        "gradient_mae": gradient_mae(
            reference,
            prediction
        ),
    }


# =========================================================
# データ読み込み
# =========================================================

def load_test_volumes(path):

    archive = np.load(path)

    key = (
        "Test"
        if "Test" in archive.files
        else archive.files[0]
    )

    volumes = archive[key]

    if volumes.ndim != 4:

        raise ValueError(
            f"想定外のテストデータ形状: "
            f"{volumes.shape}"
        )

    # (D,H,W,N)
    # ↓
    # (N,D,H,W)

    volumes = np.transpose(
        volumes,
        (3, 0, 1, 2)
    )

    if tuple(
        volumes.shape[1:]
    ) != FULL_SHAPE:

        raise ValueError(
            f"フル解像度 "
            f"{FULL_SHAPE} "
            f"が必要です。"
            f"実際: "
            f"{volumes.shape[1:]}"
        )

    return (
        volumes.astype(
            np.float32
        ),
        key
    )


def load_single_volume(path):

    if (
        path.suffix.lower()
        == ".npy"
    ):

        volume = np.load(path)

    elif (
        path.suffix.lower()
        == ".npz"
    ):

        archive = np.load(path)

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
            f"未対応の画像形式です: "
            f"{path}"
        )

    volume = np.squeeze(
        volume
    )

    if tuple(
        volume.shape
    ) != FULL_SHAPE:

        raise ValueError(
            f"{path.name} の形状が "
            f"{FULL_SHAPE} "
            f"ではありません: "
            f"{volume.shape}"
        )

    return volume.astype(
        np.float32
    )


def load_test_pair_folders(root):

    if not root.is_dir():

        raise FileNotFoundError(
            f"テストペアフォルダが"
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

        elif len(files) == 2:

            moving_path, fixed_path = files

            print(
                f"[注意] "
                f"{pair_dir.name}: "
                f"名前順で使用: "
                f"{moving_path.name} "
                f"→ "
                f"{fixed_path.name}"
            )

        else:

            print(
                f"[スキップ] "
                f"{pair_dir.name}: "
                f"Moving/Fixedを"
                f"特定できません"
            )

            continue

        cases.append(
            (
                pair_dir.name,
                load_single_volume(
                    moving_path
                ),
                load_single_volume(
                    fixed_path
                ),
            )
        )

    if not cases:

        raise FileNotFoundError(
            f"{root.resolve()} "
            f"内に評価できるペアが"
            f"ありません。"
        )

    return cases


# =========================================================
# CT比較画像
# =========================================================

def save_case_figures(
    case_id,
    case_name,
    moving,
    fixed,
    moved_128,
    moved_wavelet,
):

    case_dir = (
        OUTPUT_DIR
        / "ct_visual_comparison"
        / f"case_{case_id:03d}_{case_name}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for slice_idx in SLICE_INDICES:

        if (
            slice_idx
            >= moving.shape[2]
        ):

            print(
                f"[スキップ] "
                f"{case_name}: "
                f"slice {slice_idx} "
                f"は範囲外"
            )

            continue

        moving_slice = (
            moving[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        fixed_slice = (
            fixed[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        moved_128_slice = (
            moved_128[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        moved_wavelet_slice = (
            moved_wavelet[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        # MovingとFixedから
        # 共通の表示濃度を決める

        display_values = torch.cat(
            [
                moving_slice.flatten(),
                fixed_slice.flatten(),
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

        figure, axes = plt.subplots(
            2,
            3,
            figsize=(13, 8),
            constrained_layout=True
        )

        images = [
            moving_slice,
            fixed_slice,
            moved_128_slice,
            moved_wavelet_slice,
        ]

        titles = [
            "Moving",
            "Fixed",
            "128 baseline: moved",
            "256 + Wavelet: moved",
        ]

        for (
            axis,
            image,
            title
        ) in zip(
            axes.flat[:4],
            images,
            titles
        ):

            axis.imshow(
                image,
                cmap="gray",
                vmin=vmin,
                vmax=vmax
            )

            axis.set_title(
                title
            )

            axis.axis(
                "off"
            )

        # ---------------------------------
        # Difference
        # ---------------------------------

        difference_128 = (
            moved_128_slice
            - fixed_slice
        )

        difference_wavelet = (
            moved_wavelet_slice
            - fixed_slice
        )

        difference_limit = max(
            torch.quantile(
                difference_128.abs(),
                0.99
            ).item(),

            torch.quantile(
                difference_wavelet.abs(),
                0.99
            ).item(),

            1e-6,
        )

        difference_data = [
            (
                difference_128,
                "128 baseline: moved − fixed"
            ),
            (
                difference_wavelet,
                "256 + Wavelet: moved − fixed"
            ),
        ]

        for (
            axis,
            (
                difference,
                title
            )
        ) in zip(
            axes.flat[4:],
            difference_data
        ):

            im = axis.imshow(
                difference,
                cmap="coolwarm",
                vmin=-difference_limit,
                vmax=difference_limit
            )

            axis.set_title(
                title
            )

            axis.axis(
                "off"
            )

            figure.colorbar(
                im,
                ax=axis,
                fraction=0.046,
                pad=0.04
            )

        figure.suptitle(
            f"{case_name} "
            f"— Slice "
            f"{slice_idx}",
            fontsize=16
        )

        save_path = (
            case_dir
            /
            f"slice_{slice_idx:03d}_comparison.png"
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
            f"保存: "
            f"{save_path}"
        )


# =========================================================
# 5スライスを1枚にまとめる
# =========================================================

def save_slice_overview(
    case_id,
    case_name,
    moving,
    fixed,
    moved_128,
    moved_wavelet,
):

    valid_slices = [
        s
        for s in SLICE_INDICES
        if s < moving.shape[2]
    ]

    case_dir = (
        OUTPUT_DIR
        / "ct_visual_comparison"
        / f"case_{case_id:03d}_{case_name}"
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

        moving_slice = (
            moving[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        fixed_slice = (
            fixed[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        moved_128_slice = (
            moved_128[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        moved_wavelet_slice = (
            moved_wavelet[
                0,
                0,
                slice_idx
            ]
            .detach()
            .cpu()
        )

        display_values = torch.cat(
            [
                moving_slice.flatten(),
                fixed_slice.flatten(),
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

        images = [
            moving_slice,
            fixed_slice,
            moved_128_slice,
            moved_wavelet_slice,
        ]

        titles = [
            "Moving",
            "Fixed",
            "128 baseline",
            "256 + Wavelet",
        ]

        for col_idx, (
            image,
            title
        ) in enumerate(
            zip(
                images,
                titles
            )
        ):

            ax = axes[
                row_idx,
                col_idx
            ]

            ax.imshow(
                image,
                cmap="gray",
                vmin=vmin,
                vmax=vmax
            )

            if row_idx == 0:
                ax.set_title(
                    title
                )

            if col_idx == 0:
                ax.set_ylabel(
                    f"Slice "
                    f"{slice_idx}",
                    fontsize=12
                )

            ax.set_xticks([])
            ax.set_yticks([])

    figure.suptitle(
        f"{case_name}: "
        f"128 vs 256 + Wavelet",
        fontsize=18
    )

    save_path = (
        case_dir
        /
        "slice_overview.png"
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
        f"保存: "
        f"{save_path}"
    )


# =========================================================
# main
# =========================================================

def main():

    required_paths = (
        CHECKPOINT_128_PATH,
        CHECKPOINT_WAVELET_PATH,
    )

    if TEST_DATA_PATH is not None:

        required_paths = (
            TEST_DATA_PATH,
        ) + required_paths

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"ファイルが"
                f"見つかりません: "
                f"{path.resolve()}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------
    # Test data
    # ---------------------------------

    if TEST_DATA_PATH is None:

        test_cases = (
            load_test_pair_folders(
                TEST_PAIR_ROOT
            )
        )

        test_source = (
            f"pair folders: "
            f"{TEST_PAIR_ROOT.resolve()}"
        )

    else:

        volumes, array_key = (
            load_test_volumes(
                TEST_DATA_PATH
            )
        )

        test_cases = [
            (
                f"pair_"
                f"{moving_id}_"
                f"{fixed_id}",
                volumes[
                    moving_id
                ],
                volumes[
                    fixed_id
                ],
            )
            for (
                moving_id,
                fixed_id
            )
            in TEST_PAIRS
        ]

        test_source = (
            f"npz key: "
            f"{array_key}"
        )

    # ---------------------------------
    # Model 128
    # ---------------------------------

    model_128 = (
        vxm.networks.VxmDense(
            LOW_SHAPE,
            NB_FEATURES,
            int_steps=0
        )
        .to(
            DEVICE
        )
    )

    model_128.load_state_dict(
        state_dict(
            CHECKPOINT_128_PATH
        )
    )

    model_128.eval()

    # ---------------------------------
    # Model Wavelet
    # ---------------------------------

    model_wavelet = (
        vxm.networks
        .VxmDense_128_256_256(
            FULL_SHAPE,
            NB_FEATURES,
            int_steps=0
        )
        .to(
            DEVICE
        )
    )

    model_wavelet.load_state_dict(
        state_dict(
            CHECKPOINT_WAVELET_PATH
        )
    )

    model_wavelet.eval()

    # ---------------------------------
    # Transformer
    # ---------------------------------

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

    # ---------------------------------
    # Wavelet
    # ---------------------------------

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

    rows = []

    # =====================================================
    # Evaluation
    # =====================================================

    with torch.no_grad():

        for case_id, (
            case_name,
            moving_volume,
            fixed_volume
        ) in enumerate(
            test_cases
        ):

            print(
                "\n"
                "===================================="
            )

            print(
                f"Case "
                f"{case_id}: "
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

            fixed = (
                torch.from_numpy(
                    fixed_volume[
                        None,
                        None
                    ]
                )
                .to(
                    DEVICE
                )
            )

            # =================================================
            # 128 baseline
            # =================================================

            moving_low = F.interpolate(
                moving,
                size=LOW_SHAPE,
                mode="trilinear",
                align_corners=False
            )

            fixed_low = F.interpolate(
                fixed,
                size=LOW_SHAPE,
                mode="trilinear",
                align_corners=False
            )

            _, flow_128_low = model_128(
                moving_low,
                fixed_low
            )

            flow_128_full = F.interpolate(
                flow_128_low,
                size=FULL_SHAPE,
                mode="trilinear",
                align_corners=False
            ) * 2.0

            moved_128 = (
                full_transformer(
                    moving,
                    flow_128_full
                )
            )

            # =================================================
            # 256 + Wavelet
            # =================================================

            moving_bands = (
                analysis(
                    moving
                )[
                    :,
                    :,
                    ::2,
                    ::2,
                    ::2
                ]
            )

            fixed_bands = (
                analysis(
                    fixed
                )[
                    :,
                    :,
                    ::2,
                    ::2,
                    ::2
                ]
            )

            flow_wavelet_low = (
                model_wavelet(
                    moving_bands,
                    fixed_bands
                )
            )

            warped_bands = (
                torch.cat(
                    [
                        low_transformer(
                            moving_bands[
                                :,
                                channel:
                                channel + 1
                            ],
                            flow_wavelet_low
                        )
                        for channel
                        in range(8)
                    ],
                    dim=1
                )
            )

            moved_wavelet = (
                synthesize(
                    warped_bands,
                    synthesis_filters
                )
            )

            # =================================================
            # CT可視化
            # =================================================

            save_case_figures(
                case_id,
                case_name,
                moving,
                fixed,
                moved_128,
                moved_wavelet
            )

            save_slice_overview(
                case_id,
                case_name,
                moving,
                fixed,
                moved_128,
                moved_wavelet
            )

            # =================================================
            # Metrics
            # =================================================

            for method, moved in (
                (
                    "128_baseline",
                    moved_128
                ),
                (
                    "256_wavelet",
                    moved_wavelet
                ),
            ):

                row = {
                    "case": case_id,
                    "case_name": case_name,
                    "method": method,
                }

                row.update(
                    metrics(
                        fixed,
                        moved
                    )
                )

                rows.append(
                    row
                )

    # =====================================================
    # CSV
    # =====================================================

    csv_path = (
        OUTPUT_DIR
        /
        "test_pair_registration_metrics.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # =====================================================
    # Boxplot
    # =====================================================

    methods = (
        "128_baseline",
        "256_wavelet",
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
        constrained_layout=True
    )

    for (
        axis,
        metric_name
    ) in zip(
        axes,
        (
            "rmse",
            "ncc",
            "gradient_mae"
        )
    ):

        values = [
            [
                row[
                    metric_name
                ]
                for row
                in rows
                if (
                    row[
                        "method"
                    ]
                    == method
                )
            ]
            for method
            in methods
        ]

        axis.boxplot(
            values,
            tick_labels=[
                "128",
                "256 + Wavelet"
            ]
        )

        axis.set_title(
            metric_name.upper()
        )

        axis.grid(
            axis="y",
            alpha=0.3
        )

    metric_figure_path = (
        OUTPUT_DIR
        /
        "test_pair_metric_comparison.png"
    )

    figure.savefig(
        metric_figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================
    # Finish
    # =====================================================

    print(
        "\n評価完了"
    )

    print(
        f"テストデータ: "
        f"{test_source}"
    )

    print(
        f"CSV: "
        f"{csv_path.resolve()}"
    )

    print(
        f"箱ひげ図: "
        f"{metric_figure_path.resolve()}"
    )

    print(
        "CT画像:"
    )

    print(
        (
            OUTPUT_DIR
            /
            "ct_visual_comparison"
        ).resolve()
    )


if __name__ == "__main__":
    main()