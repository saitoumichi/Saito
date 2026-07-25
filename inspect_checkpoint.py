#!/usr/bin/env python3
"""Inspect a PyTorch checkpoint for 80k, wavelet, and curriculum evidence.

Usage:
    python inspect_checkpoint.py /path/to/model_analysis_pipeline_pretrain.pth

This script does not modify the checkpoint.  A checkpoint containing only a
state_dict cannot prove how it was trained; the script explicitly reports that
case instead of guessing from the filename.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch が見つかりません。学習に使用した Jupyter/conda 環境で実行してください。\n"
        "例: conda activate <training-env>"
    ) from exc


def load_checkpoint(path: Path) -> Any:
    """Load weights without unpickling arbitrary objects where supported."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch does not accept weights_only.
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"weights_only=True では読めませんでした: {exc}")
        print("注意: 以下は信頼できるチェックポイントに対してのみ実行してください。")
        return torch.load(path, map_location="cpu", weights_only=False)


def flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Return leaf values, retaining paths such as training.curriculum.enabled."""
    if not isinstance(value, Mapping):
        return [(prefix, value)]
    leaves: list[tuple[str, Any]] = []
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        # Parameters are not training metadata; traversing them is needlessly slow.
        if str(key) in {"model_state_dict", "state_dict", "optimizer_state_dict"}:
            continue
        leaves.extend(flatten(child, child_prefix))
    return leaves


def has_term(items: list[tuple[str, Any]], terms: tuple[str, ...]) -> list[str]:
    matches = []
    for key, value in items:
        if isinstance(value, torch.Tensor):
            continue
        text = f"{key}={value}".lower()
        if any(term in text for term in terms):
            matches.append(f"{key}: {value}")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="確認する .pth ファイル")
    args = parser.parse_args()

    path = args.checkpoint.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"ファイルが見つかりません: {path}")

    print(f"checkpoint: {path}")
    print(f"size: {path.stat().st_size / (1024 ** 2):.1f} MiB\n")
    checkpoint = load_checkpoint(path)

    if not isinstance(checkpoint, Mapping):
        print(f"形式: {type(checkpoint).__name__}（state_dict 単体の可能性があります）")
        print("判定: 保存された学習条件がないため、80k／Wavelet／カリキュラム学習は確認不能です。")
        return

    keys = list(checkpoint.keys())
    print("トップレベルキー:")
    print("  " + ", ".join(map(str, keys)))

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if isinstance(state_dict, Mapping):
        parameter_keys = list(state_dict.keys())
        print(f"\nモデル重み: {len(parameter_keys)} tensor entries")
        wavelet_weight_keys = [key for key in parameter_keys if "wavelet" in key.lower() or "haar" in key.lower()]
        if wavelet_weight_keys:
            print("Wavelet/Haar を示す重み名:")
            print("  " + "\n  ".join(wavelet_weight_keys[:20]))
        else:
            print("Wavelet/Haar を示す重み名: なし（前処理がモデル外なら、これは正常です）")

    items = flatten(checkpoint)
    epoch_matches = has_term(items, ("epoch", "step", "iteration", "iter"))
    wavelet_matches = has_term(items, ("wavelet", "haar", "dwt"))
    curriculum_matches = has_term(items, ("curriculum", "difficulty", "stage", "schedule"))

    def show(label: str, matches: list[str]) -> None:
        print(f"\n{label}:")
        if matches:
            print("  " + "\n  ".join(matches[:30]))
        else:
            print("  保存メタデータには見つかりません")

    show("80k を確認するための epoch/step 情報", epoch_matches)
    show("Wavelet 使用を示す保存メタデータ", wavelet_matches)
    show("カリキュラム学習を示す保存メタデータ", curriculum_matches)

    print("\n結論:")
    if curriculum_matches:
        print("  カリキュラム学習に関するメタデータがあります。値が意図した設定か確認してください。")
    else:
        print("  カリキュラム学習の確証はありません（重みのみでは学習手法を判定できません）。")
    if wavelet_matches:
        print("  Wavelet に関するメタデータがあります。")
    else:
        print("  Wavelet はモデル外の前処理なら checkpoint に現れないため、学習ノートブックも照合してください。")


if __name__ == "__main__":
    main()
