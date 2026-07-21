#!/usr/bin/env python3

from __future__ import annotations

import ast
import struct
import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def read_npy_header(npz_path: Path) -> None:
    with zipfile.ZipFile(npz_path, "r") as zf:
        members = zf.namelist()
        if not members:
            print("NPZの中身が空です。")
            return

        print(f"file: {npz_path}")
        print("members:")

        for name in members:
            with zf.open(name) as f:
                magic = f.read(6)
                if magic != b"\x93NUMPY":
                    print(f"  - {name}: NPY形式ではありません")
                    continue

                major = f.read(1)[0]
                _minor = f.read(1)[0]

                if major == 1:
                    header_len = struct.unpack("<H", f.read(2))[0]
                else:
                    header_len = struct.unpack("<I", f.read(4))[0]

                header = f.read(header_len).decode("latin1")
                meta = ast.literal_eval(header)
                key_name = Path(name).stem

                print(
                    f"  - key={key_name}, shape={meta['shape']}, "
                    f"dtype={meta['descr']}, fortran_order={meta['fortran_order']}"
                )


def inspect_array(npz_path: Path, key_name: str = "Train") -> None:
    if np is None:
        print("numpy がないため、配列の統計確認はできません。")
        return

    with np.load(npz_path) as data:
        if key_name not in data:
            print(f"キー '{key_name}' は見つかりません。利用可能なキー: {list(data.keys())}")
            return

        arr = data[key_name]
        print()
        print(f"inspect key: {key_name}")
        print(f"shape: {arr.shape}")
        print(f"dtype: {arr.dtype}")
        print(f"min: {arr.min()}")
        print(f"max: {arr.max()}")
        print(f"mean: {arr.mean()}")
        print(f"std: {arr.std()}")

        if arr.ndim == 4:
            transposed = np.transpose(arr, (3, 0, 1, 2))
            print(f"transposed shape (notebook format): {transposed.shape}")

            sample_idx = 0
            slice_idx = transposed.shape[1] // 2
            sample = transposed[sample_idx]
            center_slice = sample[slice_idx]

            print(f"sample[0] min: {sample.min()}")
            print(f"sample[0] max: {sample.max()}")
            print(f"center slice index: {slice_idx}")
            print("center slice preview:")
            preview = center_slice[::16, ::16]
            print(preview)

            if plt is not None:
                out_path = npz_path.with_name(f"{npz_path.stem}_{key_name}_slice.png")
                plt.figure(figsize=(6, 6))
                plt.imshow(center_slice, cmap="gray")
                plt.title(f"{key_name} sample0 slice{slice_idx}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"slice image saved: {out_path}")
            else:
                print("matplotlib がないため、スライス画像の保存はスキップしました。")
        else:
            print("4次元配列ではないため、ノートブック形式への transpose と画像保存はスキップしました。")


def main() -> int:
    if len(sys.argv) > 1:
        npz_path = Path(sys.argv[1]).expanduser()
    else:
        npz_path = Path("Data/TrainData_NoBed.npz")

    if not npz_path.exists():
        print(f"ファイルが見つかりません: {npz_path}")
        print("使い方: python3 inspect_npz.py /path/to/file.npz")
        return 1

    if npz_path.suffix.lower() != ".npz":
        print(f"NPZファイルを指定してください: {npz_path}")
        return 1

    read_npy_header(npz_path)
    inspect_array(npz_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
