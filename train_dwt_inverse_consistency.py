"""DWT + inverse-consistency VoxelMorph training.

既存の Copy1 と同じ1段3D Haar DWTを使用する。
事前学習では人工DVFによる順方向の教師ありDVF損失を残し、
逆方向には画像損失と双方向の逆整合性損失を加える。
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm


SAITO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SAITO_DIR.parent))
import voxelmorph as vxm  # noqa: E402


# -------------------- settings --------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = SAITO_DIR / "Data" / "TrainData_NoBed.npz"
OUTPUT_DIR = SAITO_DIR / "dwt_inverse_consistency_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PRETRAIN_EPOCHS = 80_000
FINETUNE_EPOCHS = 30_000
BATCH_SIZE = 2
PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-5
SAVE_EVERY = 2_000

# 損失の重み。まずは同程度のオーダーから開始し、学習曲線を見て調整する。
IMAGE_WEIGHT = 100.0
FORWARD_DVF_WEIGHT = 0.01
INVERSE_CONSISTENCY_WEIGHT = 1.0

NB_FEATURES = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16],
]


class Haar3DAnalysis(nn.Module):
    def __init__(self):
        super().__init__()
        low = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
        high = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
        filters = []
        for z in (low, high):
            for y in (low, high):
                for x in (low, high):
                    filters.append(z[:, None, None] * y[None, :, None] * x[None, None, :])
        self.register_buffer("weight", torch.stack(filters).unsqueeze(1).float())

    def forward(self, image):
        return F.conv3d(F.pad(image, (0, 1, 0, 1, 0, 1)), self.weight)


def dwt3d(image, analysis):
    return analysis(image)[:, :, ::2, ::2, ::2]


def make_synthesis_filters(device):
    low = torch.tensor([1.0, 1.0], device=device) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], device=device) / math.sqrt(2.0)
    filters = []
    for z in (low, high):
        for y in (low, high):
            for x in (low, high):
                filters.append(z[:, None, None] * y[None, :, None] * x[None, None, :])
    return torch.flip(torch.stack(filters), dims=[1, 2, 3]).unsqueeze(1)


def idwt3d(coefficients, synthesis_filters):
    batch, channels, depth, height, width = coefficients.shape
    upsampled = torch.zeros(
        batch, channels, depth * 2, height * 2, width * 2,
        dtype=coefficients.dtype, device=coefficients.device,
    )
    upsampled[:, :, ::2, ::2, ::2] = coefficients
    bands = []
    for index in range(channels):
        filtered = F.conv3d(upsampled[:, index:index + 1], synthesis_filters[index:index + 1], padding=1)
        bands.append(filtered[:, :, :depth * 2, :height * 2, :width * 2])
    return torch.cat(bands, dim=1).sum(dim=1, keepdim=True)


def data_generator(volumes, batch_size):
    """MovingとFixedに同じ症例を選ばない既存Copy1と同じサンプリング。"""
    while True:
        source_indices = np.random.randint(0, len(volumes), size=batch_size)
        target_indices = np.random.randint(0, len(volumes), size=batch_size)
        while np.any(source_indices == target_indices):
            same = source_indices == target_indices
            target_indices[same] = np.random.randint(0, len(volumes), size=same.sum())
        source = torch.from_numpy(volumes[source_indices]).unsqueeze(1).float()
        target = torch.from_numpy(volumes[target_indices]).unsqueeze(1).float()
        yield source.to(DEVICE), target.to(DEVICE)


def smooth_random_flow(batch_size, shift_range):
    coarse = (torch.rand((batch_size, 3, 8, 16, 16)) * 2.0 - 1.0) * shift_range
    smoothed = gaussian_filter(coarse.numpy(), sigma=[0, 0, 2.0, 2.0, 2.0])
    full = F.interpolate(torch.from_numpy(smoothed).float(), size=(128, 256, 256), mode="trilinear", align_corners=False)
    low = F.interpolate(full, size=(64, 128, 128), mode="trilinear", align_corners=False)
    return full.to(DEVICE), low.to(DEVICE)


def compose_flow(first_flow, second_flow, lowres_transformer):
    """first_flow の後に second_flow を適用した合成DVF。

    VoxelMorphのDVFはボクセル単位であるため、grid_sampleの正規化座標を
    手実装せず、既存SpatialTransformerでsecond_flowをサンプルする。
    """
    return first_flow + lowres_transformer(second_flow, first_flow)


def inverse_consistency_loss(flow_ab, flow_ba, lowres_transformer):
    cycle_on_b = compose_flow(flow_ab, flow_ba, lowres_transformer)
    cycle_on_a = compose_flow(flow_ba, flow_ab, lowres_transformer)
    return 0.5 * (torch.mean(cycle_on_b ** 2) + torch.mean(cycle_on_a ** 2))


def dwt_register(model, source, target, analysis, synthesis_filters, lowres_transformer):
    source_w = dwt3d(source, analysis)
    target_w = dwt3d(target, analysis)
    flow = model(source_w, target_w)
    moved = idwt3d(lowres_transformer(source_w, flow), synthesis_filters)
    return moved, flow


def save_checkpoint(path, epoch, model, optimizer, phase):
    torch.save(
        {
            "phase": phase,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main():
    print(f"device: {DEVICE}")
    volumes = np.transpose(np.load(DATA_PATH)["Train"], (3, 0, 1, 2)).astype(np.float32)
    batches = data_generator(volumes, BATCH_SIZE)

    analysis = Haar3DAnalysis().to(DEVICE)
    synthesis_filters = make_synthesis_filters(DEVICE)
    lowres_transformer = vxm.layers.SpatialTransformer((64, 128, 128)).to(DEVICE)
    fullres_transformer = vxm.layers.SpatialTransformer((128, 256, 256)).to(DEVICE)
    model = vxm.networks.VxmDense_128_256_256((128, 256, 256), NB_FEATURES, int_steps=0).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    # -------- pretraining: synthetic 3D DVF --------
    for epoch in tqdm(range(1, PRETRAIN_EPOCHS + 1), desc="DWT inverse-consistency pretraining"):
        shift_range = 1 + (epoch - 1) // 2_000
        moving, _ = next(batches)
        true_flow_full, true_flow_low = smooth_random_flow(BATCH_SIZE, shift_range)
        moving_prime = fullres_transformer(moving, true_flow_full)

        optimizer.zero_grad()
        moved_ab, flow_ab = dwt_register(model, moving, moving_prime, analysis, synthesis_filters, lowres_transformer)
        moved_ba, flow_ba = dwt_register(model, moving_prime, moving, analysis, synthesis_filters, lowres_transformer)

        # B→Aには一般に -true_flow ではないため、誤った教師DVF損失は置かない。
        loss_image_ab = F.mse_loss(moved_ab, moving_prime)
        loss_image_ba = F.mse_loss(moved_ba, moving)
        loss_dvf_ab = F.mse_loss(flow_ab, true_flow_low)
        loss_inverse = inverse_consistency_loss(flow_ab, flow_ba, lowres_transformer)
        loss = (
            IMAGE_WEIGHT * (loss_image_ab + loss_image_ba)
            + FORWARD_DVF_WEIGHT * loss_dvf_ab
            + INVERSE_CONSISTENCY_WEIGHT * loss_inverse
        )
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"epoch {epoch}/{PRETRAIN_EPOCHS} | total={loss.item():.5f} | "
                f"image_ab={loss_image_ab.item():.5f} | image_ba={loss_image_ba.item():.5f} | "
                f"dvf_ab={loss_dvf_ab.item():.5f} | inverse={loss_inverse.item():.5f} | ±{shift_range}"
            )
        if epoch % SAVE_EVERY == 0 or epoch == PRETRAIN_EPOCHS:
            save_checkpoint(OUTPUT_DIR / f"pretrain_epoch_{epoch:05d}.pth", epoch, model, optimizer, "pretrain")

    # -------- fine-tuning: real pairs, bidirectional image + inverse loss --------
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)
    for epoch in tqdm(range(1, FINETUNE_EPOCHS + 1), desc="DWT inverse-consistency fine-tuning"):
        moving, fixed = next(batches)
        optimizer.zero_grad()
        moved_ab, flow_ab = dwt_register(model, moving, fixed, analysis, synthesis_filters, lowres_transformer)
        moved_ba, flow_ba = dwt_register(model, fixed, moving, analysis, synthesis_filters, lowres_transformer)
        loss_image_ab = F.mse_loss(moved_ab, fixed)
        loss_image_ba = F.mse_loss(moved_ba, moving)
        loss_inverse = inverse_consistency_loss(flow_ab, flow_ba, lowres_transformer)
        loss = IMAGE_WEIGHT * (loss_image_ab + loss_image_ba) + INVERSE_CONSISTENCY_WEIGHT * loss_inverse
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"fine-tune {epoch}/{FINETUNE_EPOCHS} | total={loss.item():.5f} | "
                f"image_ab={loss_image_ab.item():.5f} | image_ba={loss_image_ba.item():.5f} | inverse={loss_inverse.item():.5f}"
            )
        if epoch % SAVE_EVERY == 0 or epoch == FINETUNE_EPOCHS:
            save_checkpoint(OUTPUT_DIR / f"finetune_epoch_{epoch:05d}.pth", epoch, model, optimizer, "finetune")


if __name__ == "__main__":
    main()
