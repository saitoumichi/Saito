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

# ===== ユーザー設定 =====
# 単一npzを使う場合だけ設定する。pair別フォルダを使う場合は None のままにする。
TEST_DATA_PATH = None
TEST_PAIR_ROOT = Path("Data/TestData")
CHECKPOINT_128_PATH = Path("yamatoCode/a.pth")
CHECKPOINT_WAVELET_PATH = Path(
    "curriculum_checkpoints_dvf_scale_corrected/pretrain_epoch_80000.pth"
)
# TEST_DATA_PATH を使う場合だけ、対応するMoving / Fixedの症例番号に変更すること。
TEST_PAIRS = [(0, 1), (2, 3), (4, 5)]
OUTPUT_DIR = Path("comparison_128_vs_256_wavelet")
# =======================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FULL_SHAPE = (128, 256, 256)
LOW_SHAPE = (64, 128, 128)
NB_FEATURES = [[32, 64, 64, 64, 64], [64, 64, 64, 64, 64, 32, 16, 16]]


class Haar3DAnalysisOnly(nn.Module):
    def __init__(self):
        super().__init__()
        low = torch.tensor([1.0, 1.0]) / np.sqrt(2.0)
        high = torch.tensor([1.0, -1.0]) / np.sqrt(2.0)
        filters = [
            z[:, None, None] * y[None, :, None] * x[None, None, :]
            for z in (low, high) for y in (low, high) for x in (low, high)
        ]
        self.register_buffer("weight", torch.stack(filters).unsqueeze(1).float())

    def forward(self, image):
        return F.conv3d(F.pad(image, (0, 1, 0, 1, 0, 1)), self.weight, stride=1)


def create_synthesis_filters(device):
    low = torch.tensor([1.0, 1.0], device=device) / np.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], device=device) / np.sqrt(2.0)
    filters = [
        z[:, None, None] * y[None, :, None] * x[None, None, :]
        for z in (low, high) for y in (low, high) for x in (low, high)
    ]
    return torch.flip(torch.stack(filters), dims=[1, 2, 3]).unsqueeze(1)


def synthesize(wavelet_bands, filters):
    up = torch.zeros(
        wavelet_bands.shape[0], wavelet_bands.shape[1],
        wavelet_bands.shape[2] * 2, wavelet_bands.shape[3] * 2, wavelet_bands.shape[4] * 2,
        device=wavelet_bands.device, dtype=wavelet_bands.dtype,
    )
    up[:, :, ::2, ::2, ::2] = wavelet_bands
    reconstructed = []
    for band in range(8):
        filtered = F.conv3d(up[:, band:band + 1], filters[band:band + 1], padding=1)
        reconstructed.append(filtered[:, :, :up.shape[2], :up.shape[3], :up.shape[4]])
    return torch.cat(reconstructed, dim=1).sum(dim=1, keepdim=True)


def state_dict(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    return checkpoint.get("model_state_dict", checkpoint)


def ncc(reference, prediction, eps=1e-8):
    reference = reference - reference.mean()
    prediction = prediction - prediction.mean()
    return ((reference * prediction).sum() / torch.sqrt(
        reference.square().sum() * prediction.square().sum() + eps
    )).item()


def gradient_mae(reference, prediction):
    errors = []
    for axis in (2, 3, 4):
        reference_gradient = torch.diff(reference, dim=axis)
        prediction_gradient = torch.diff(prediction, dim=axis)
        errors.append((reference_gradient - prediction_gradient).abs().mean())
    return torch.stack(errors).mean().item()


def metrics(reference, prediction):
    mse = torch.mean((reference - prediction) ** 2).item()
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "ncc": ncc(reference, prediction),
        "gradient_mae": gradient_mae(reference, prediction),
    }


def load_test_volumes(path):
    archive = np.load(path)
    key = "Test" if "Test" in archive.files else archive.files[0]
    volumes = archive[key]
    if volumes.ndim != 4:
        raise ValueError(f"想定外のテストデータ形状: {volumes.shape}")
    # (D, H, W, N) を (N, D, H, W) へ変換する既存ノートブックの形式。
    volumes = np.transpose(volumes, (3, 0, 1, 2))
    if tuple(volumes.shape[1:]) != FULL_SHAPE:
        raise ValueError(f"フル解像度 {FULL_SHAPE} が必要です。実際: {volumes.shape[1:]}")
    return volumes.astype(np.float32), key


def load_single_volume(path):
    """npyまたはnpzから1症例の3D画像を読み込む。"""
    if path.suffix.lower() == ".npy":
        volume = np.load(path)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(
                f"{path} は複数配列を含みます。Moving/Fixedを別ファイルにするか、"
                "moving/fixedキーを持つ1つのnpzにしてください。"
            )
        volume = archive[archive.files[0]]
    else:
        raise ValueError(f"未対応の画像形式です: {path}")
    volume = np.squeeze(volume)
    if tuple(volume.shape) != FULL_SHAPE:
        raise ValueError(f"{path.name} の形状が {FULL_SHAPE} ではありません: {volume.shape}")
    return volume.astype(np.float32)


def load_test_pair_folders(root):
    """pair1等の各フォルダからMoving/Fixedの3D画像を読み込む。"""
    if not root.is_dir():
        raise FileNotFoundError(f"テストペアフォルダが見つかりません: {root.resolve()}")
    cases = []
    for pair_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            list(pair_dir.glob("*.npy")) + list(pair_dir.glob("*.npz"))
        )
        # 実データでは registered_masked_A_* をMoving、fixed_masked_B_* をFixedとして扱う。
        moving_files = [
            path for path in files
            if "moving" in path.stem.lower() or "registered" in path.stem.lower()
        ]
        fixed_files = [path for path in files if "fixed" in path.stem.lower()]
        if moving_files and fixed_files:
            moving_path, fixed_path = moving_files[0], fixed_files[0]
        elif len(files) == 2:
            # ファイル名にMoving/Fixedがない場合は、名前順の1番目をMoving、2番目をFixedとみなす。
            moving_path, fixed_path = files
            print(f"[注意] {pair_dir.name}: Moving/Fixed名がないため名前順で使用: "
                  f"{moving_path.name} → {fixed_path.name}")
        else:
            print(f"[スキップ] {pair_dir.name}: npy/npzのMoving・Fixedペアを特定できません")
            continue
        cases.append((pair_dir.name, load_single_volume(moving_path), load_single_volume(fixed_path)))
    if not cases:
        raise FileNotFoundError(
            f"{root.resolve()} 内に評価できるペアがありません。各pairフォルダに"
            "Moving/Fixedを名前に含む.npyまたは.npzを置いてください。"
        )
    return cases


def save_case_figure(case_id, moving, fixed, moved_128, moved_wavelet):
    middle = moving.shape[2] // 2
    images = [moving, fixed, moved_128, moved_wavelet]
    titles = ["Moving", "Fixed", "128 baseline", "256 + Wavelet"]
    vmin, vmax = np.percentile(moving[0, 0, middle].cpu(), [1, 99])
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image[0, 0, middle].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(OUTPUT_DIR / f"case_{case_id:02d}_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    required_paths = (CHECKPOINT_128_PATH, CHECKPOINT_WAVELET_PATH)
    if TEST_DATA_PATH is not None:
        required_paths = (TEST_DATA_PATH,) + required_paths
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path.resolve()}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    if TEST_DATA_PATH is None:
        test_cases = load_test_pair_folders(TEST_PAIR_ROOT)
        test_source = f"pair folders: {TEST_PAIR_ROOT.resolve()}"
    else:
        volumes, array_key = load_test_volumes(TEST_DATA_PATH)
        test_cases = [
            (f"pair_{moving_id}_{fixed_id}", volumes[moving_id], volumes[fixed_id])
            for moving_id, fixed_id in TEST_PAIRS
        ]
        test_source = f"npz key: {array_key}"

    model_128 = vxm.networks.VxmDense(LOW_SHAPE, NB_FEATURES, int_steps=0).to(DEVICE)
    model_128.load_state_dict(state_dict(CHECKPOINT_128_PATH))
    model_128.eval()
    model_wavelet = vxm.networks.VxmDense_128_256_256(FULL_SHAPE, NB_FEATURES, int_steps=0).to(DEVICE)
    model_wavelet.load_state_dict(state_dict(CHECKPOINT_WAVELET_PATH))
    model_wavelet.eval()

    full_transformer = vxm.layers.SpatialTransformer(FULL_SHAPE).to(DEVICE)
    low_transformer = vxm.layers.SpatialTransformer(LOW_SHAPE).to(DEVICE)
    analysis = Haar3DAnalysisOnly().to(DEVICE)
    synthesis_filters = create_synthesis_filters(DEVICE)
    rows = []

    with torch.no_grad():
        for case_id, (case_name, moving_volume, fixed_volume) in enumerate(test_cases):
            moving = torch.from_numpy(moving_volume[None, None]).to(DEVICE)
            fixed = torch.from_numpy(fixed_volume[None, None]).to(DEVICE)

            # 128ベースライン: 低解像度DVFをフル解像度へ拡大し、元の256画像を変形して公平に評価する。
            moving_low = F.interpolate(moving, size=LOW_SHAPE, mode="trilinear", align_corners=False)
            fixed_low = F.interpolate(fixed, size=LOW_SHAPE, mode="trilinear", align_corners=False)
            _, flow_128_low = model_128(moving_low, fixed_low)
            flow_128_full = F.interpolate(flow_128_low, size=FULL_SHAPE, mode="trilinear", align_corners=False) * 2.0
            moved_128 = full_transformer(moving, flow_128_full)

            # Wavelet: 8バンドを低解像度格子へ配置し、同じDVFで変形して再合成する。
            moving_bands = analysis(moving)[:, :, ::2, ::2, ::2]
            fixed_bands = analysis(fixed)[:, :, ::2, ::2, ::2]
            flow_wavelet_low = model_wavelet(moving_bands, fixed_bands)
            warped_bands = torch.cat([
                low_transformer(moving_bands[:, channel:channel + 1], flow_wavelet_low)
                for channel in range(8)
            ], dim=1)
            moved_wavelet = synthesize(warped_bands, synthesis_filters)

            save_case_figure(case_id, moving, fixed, moved_128, moved_wavelet)
            for method, moved in (("128_baseline", moved_128), ("256_wavelet", moved_wavelet)):
                row = {"case": case_id, "case_name": case_name, "method": method}
                row.update(metrics(fixed, moved))
                rows.append(row)

    csv_path = OUTPUT_DIR / "test_pair_registration_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    methods = ("128_baseline", "256_wavelet")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, metric_name in zip(axes, ("rmse", "ncc", "gradient_mae")):
        values = [[row[metric_name] for row in rows if row["method"] == method] for method in methods]
        axis.boxplot(values, tick_labels=["128", "256 + Wavelet"])
        axis.set_title(metric_name.upper())
        axis.grid(axis="y", alpha=0.3)
    figure.savefig(OUTPUT_DIR / "test_pair_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"テストデータ: {test_source}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"比較図: {(OUTPUT_DIR / 'test_pair_metric_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
