#!/usr/bin/env python3
"""Create Phase-2 Fourier curriculum variants.

The variants are designed to test whether Phase 1's apparent long-rollout
stability gain was mostly smoothing:

* mixed: keeps raw signal present throughout the curriculum with moderate
  raw-blend weights.
* residual: treats the raw-minus-lowpass residual as signal and increases its
  weight more deliberately.

Both variants write continuous fields:

    stage = lowpass + raw_weight * (raw - lowpass)

This avoids sample-wise lowpass/raw switching, which would introduce artificial
time discontinuities into one-step training pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


MIXED_STAGES: List[Tuple[int, float, str]] = [
    (4, 0.20, "train_mixed_lp004_r020"),
    (8, 0.30, "train_mixed_lp008_r030"),
    (16, 0.40, "train_mixed_lp016_r040"),
    (32, 0.50, "train_mixed_lp032_r050"),
    (64, 0.65, "train_mixed_lp064_r065"),
]

RESIDUAL_STAGES: List[Tuple[int, float, str]] = [
    (4, 0.05, "train_residual_lp004_l005"),
    (8, 0.15, "train_residual_lp008_l015"),
    (16, 0.30, "train_residual_lp016_l030"),
    (32, 0.50, "train_residual_lp032_l050"),
    (64, 0.75, "train_residual_lp064_l075"),
]

SOFT_RESIDUAL_STAGES: List[Tuple[int, float, str]] = [
    (4, 0.20, "train_residual_soft_lp004_l020"),
    (8, 0.30, "train_residual_soft_lp008_l030"),
    (16, 0.45, "train_residual_soft_lp016_l045"),
    (32, 0.60, "train_residual_soft_lp032_l060"),
    (64, 0.80, "train_residual_soft_lp064_l080"),
]


def copy_coords_and_scales(src: h5py.File, dst: h5py.File, fields: h5py.Dataset) -> None:
    timestamp = dst.create_dataset("timestamp", data=src["timestamp"][:])
    channel = dst.create_dataset("channel", data=src["channel"][:])
    lat = dst.create_dataset("lat", data=src["lat"][:])
    lon = dst.create_dataset("lon", data=src["lon"][:])
    timestamp.make_scale("timestamp")
    channel.make_scale("channel")
    lat.make_scale("lat")
    lon.make_scale("lon")
    fields.dims[0].attach_scale(timestamp)
    fields.dims[1].attach_scale(channel)
    fields.dims[2].attach_scale(lat)
    fields.dims[3].attach_scale(lon)
    for name in ["valid_mask", "nan_fill_values"]:
        if name in src:
            dst.create_dataset(name, data=src[name][:])


def _stage_sets(kind: str) -> Dict[str, List[Tuple[int, float, str]]]:
    all_sets = {"mixed": MIXED_STAGES, "residual": RESIDUAL_STAGES, "residual_soft": SOFT_RESIDUAL_STAGES}
    if kind == "all":
        return all_sets
    return {kind: all_sets[kind]}


def _write_stage(
    dataset_root: Path,
    raw_h5: h5py.File,
    cutoff: int,
    raw_weight: float,
    stage_name: str,
    compression: str,
    force: bool,
) -> Dict[str, object]:
    lp_path = dataset_root / f"train_lp{cutoff:03d}" / "train.h5"
    if not lp_path.exists():
        raise FileNotFoundError(f"Missing low-pass stage: {lp_path}")

    stage_dir = dataset_root / stage_name
    out_path = stage_dir / "train.h5"
    if stage_dir.exists() and not force:
        raise FileExistsError(f"{stage_dir} exists; pass --force to overwrite")
    stage_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    comp = None if compression == "none" else compression
    compression_opts = 2 if compression == "gzip" else None
    raw = raw_h5["fields"]

    with h5py.File(lp_path, "r") as lp_h5, h5py.File(out_path, "w") as dst:
        lp = lp_h5["fields"]
        if lp.shape != raw.shape:
            raise ValueError(f"Shape mismatch for {lp_path}: {lp.shape} != {raw.shape}")
        fields = dst.create_dataset(
            "fields",
            shape=raw.shape,
            dtype=np.float32,
            chunks=raw.chunks,
            compression=comp,
            compression_opts=compression_opts,
            shuffle=(comp == "gzip"),
        )
        copy_coords_and_scales(raw_h5, dst, fields)
        for t0 in range(0, raw.shape[0], raw.chunks[0] if raw.chunks else 8):
            t1 = min(raw.shape[0], t0 + (raw.chunks[0] if raw.chunks else 8))
            raw_block = raw[t0:t1].astype(np.float32)
            lp_block = lp[t0:t1].astype(np.float32)
            fields[t0:t1] = (lp_block + float(raw_weight) * (raw_block - lp_block)).astype(np.float32)

    return {
        "stage": stage_name,
        "path": str(out_path),
        "lowpass_cutoff": int(cutoff),
        "raw_weight": float(raw_weight),
        "formula": "lowpass + raw_weight * (raw - lowpass)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--kind", default="all", choices=["all", "mixed", "residual", "residual_soft"])
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_path = args.dataset_root / "train_raw" / "train.h5"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    written: Dict[str, List[Dict[str, object]]] = {}
    with h5py.File(raw_path, "r") as raw_h5:
        for kind, stages in _stage_sets(args.kind).items():
            written[kind] = []
            for cutoff, raw_weight, stage_name in stages:
                written[kind].append(
                    _write_stage(
                        args.dataset_root,
                        raw_h5,
                        cutoff,
                        raw_weight,
                        stage_name,
                        args.compression,
                        args.force,
                    )
                )

    payload = {
        "dataset_root": str(args.dataset_root),
        "raw_train": str(raw_path),
        "variants": written,
        "leakage_guard": "Only training stages are transformed. valid_raw and test_raw remain raw.",
        "continuity_note": "Stages are continuous lowpass-to-raw blends rather than sample-wise lowpass/raw switches.",
    }
    manifest_path = args.dataset_root / "phase2_curriculum_variants_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
