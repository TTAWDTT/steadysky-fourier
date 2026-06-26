#!/usr/bin/env python3
"""Create cumulative low-to-high Fourier curriculum training stages.

This script reads the raw Makani training split and writes additional HDF5
training directories such as train_lp004/train.h5. The validation and test
splits remain raw. Filtering is applied only along the time axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_cutoffs(text: str) -> list[int]:
    cutoffs = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not cutoffs:
        raise ValueError("At least one cutoff is required")
    if any(x < 0 for x in cutoffs):
        raise ValueError("Cutoffs must be non-negative")
    return sorted(set(cutoffs))


def temporal_lowpass_block(data: np.ndarray, keep_modes: int) -> np.ndarray:
    coeff = np.fft.rfft(data, axis=0)
    filtered = np.zeros_like(coeff)
    stop = min(int(keep_modes) + 1, coeff.shape[0])
    filtered[:stop] = coeff[:stop]
    return np.fft.irfft(filtered, n=data.shape[0], axis=0).astype(np.float32)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cutoffs", default="2,4,8,16,32,64,128,256,512")
    parser.add_argument("--channel-block", type=int, default=1)
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cutoffs = parse_cutoffs(args.cutoffs)
    raw_path = args.dataset_root / "train_raw" / "train.h5"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    written = {}
    with h5py.File(raw_path, "r") as src:
        raw = src["fields"]
        n_time, n_channel, n_lat, n_lon = raw.shape
        max_nonzero_modes = np.fft.rfft(np.empty((n_time,), dtype=np.float32)).shape[0] - 1
        for cutoff in cutoffs:
            stage_dir = args.dataset_root / f"train_lp{cutoff:03d}"
            out_path = stage_dir / "train.h5"
            if stage_dir.exists() and not args.force:
                raise FileExistsError(f"{stage_dir} exists; pass --force to overwrite")
            stage_dir.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()

            compression = None if args.compression == "none" else args.compression
            compression_opts = 2 if args.compression == "gzip" else None
            with h5py.File(out_path, "w") as dst:
                fields = dst.create_dataset(
                    "fields",
                    shape=raw.shape,
                    dtype=np.float32,
                    chunks=raw.chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=(compression == "gzip"),
                )
                copy_coords_and_scales(src, dst, fields)
                for c0 in range(0, n_channel, args.channel_block):
                    c1 = min(n_channel, c0 + args.channel_block)
                    block = raw[:, c0:c1, :, :].astype(np.float32)
                    fields[:, c0:c1, :, :] = temporal_lowpass_block(block, cutoff)
            written[f"train_lp{cutoff:03d}"] = {
                "path": str(out_path),
                "keep_modes": int(cutoff),
                "max_nonzero_modes": int(max_nonzero_modes),
                "kept_fraction_of_nonzero_temporal_modes": float(min(cutoff, max_nonzero_modes) / max(max_nonzero_modes, 1)),
            }

    manifest_path = args.dataset_root / "fourier_curriculum_manifest.json"
    payload = {
        "dataset_root": str(args.dataset_root),
        "raw_train": str(raw_path),
        "filter_axis": "time",
        "filter": "rfft low-pass, cumulative from DC through keep_modes",
        "cutoffs": cutoffs,
        "stages": written,
        "leakage_guard": "Only train_raw is filtered. valid_raw and test_raw remain raw.",
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

