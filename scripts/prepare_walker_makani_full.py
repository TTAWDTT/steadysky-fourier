#!/usr/bin/env python3
"""Prepare full Walker 1-degree four-variable data for Makani.

Input NetCDF files are expected to contain:
  tauu_1x1.nc, tauv_1x1.nc, tos_1x1.nc, zos_1x1.nc

Output layout:
  output_root/
    train_raw/train.h5
    valid_raw/valid.h5
    test_raw/test.h5
    metadata/data.json
    stats_raw/*.npy
    manifest.json

HDF5 fields are stored as (time, channel, lat, lon), matching Makani's
MultifilesDataset default dataset_name="fields".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import xarray as xr


CHANNELS = ["tauu", "tauv", "tos", "zos"]
SPLITS = {
    "train": (0.0, 0.80),
    "valid": (0.80, 0.90),
    "test": (0.90, 1.0),
}


def _open_channel(source_root: Path, channel: str) -> xr.DataArray:
    path = source_root / f"{channel}_1x1.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path, decode_times=False)
    if channel not in ds:
        raise KeyError(f"{channel!r} not found in {path}")
    return ds[channel]


def _load_coords(source_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = _open_channel(source_root, CHANNELS[0])
    return (
        da["time"].values.astype(np.float64),
        da["lat"].values.astype(np.float32),
        da["lon"].values.astype(np.float32),
    )


def _validate_grid(source_root: Path) -> Dict[str, object]:
    ref_time, ref_lat, ref_lon = _load_coords(source_root)
    report: Dict[str, object] = {
        "channels": CHANNELS,
        "time_length": int(ref_time.shape[0]),
        "lat_length": int(ref_lat.shape[0]),
        "lon_length": int(ref_lon.shape[0]),
        "time_first_last": [float(ref_time[0]), float(ref_time[-1])],
        "lat_first_last": [float(ref_lat[0]), float(ref_lat[-1])],
        "lon_first_last": [float(ref_lon[0]), float(ref_lon[-1])],
        "time_offsets_vs_tauu_days": {},
    }
    for ch in CHANNELS:
        da = _open_channel(source_root, ch)
        if da.dims != ("time", "lat", "lon"):
            raise ValueError(f"{ch} dims must be ('time', 'lat', 'lon'), got {da.dims}")
        if da.shape != (ref_time.shape[0], ref_lat.shape[0], ref_lon.shape[0]):
            raise ValueError(f"{ch} shape mismatch: {da.shape}")
        if not np.allclose(da["lat"].values, ref_lat):
            raise ValueError(f"{ch} latitude grid mismatch")
        if not np.allclose(da["lon"].values, ref_lon):
            raise ValueError(f"{ch} longitude grid mismatch")
        dt = da["time"].values.astype(np.float64) - ref_time
        report["time_offsets_vs_tauu_days"][ch] = {
            "min": float(np.nanmin(dt)),
            "max": float(np.nanmax(dt)),
        }
        da.close()
    return report


def _split_indices(n_time: int) -> Dict[str, Tuple[int, int]]:
    return {
        name: (int(round(a * n_time)), int(round(b * n_time)))
        for name, (a, b) in SPLITS.items()
    }


def _iter_time_blocks(start: int, stop: int, block_size: int) -> Iterable[Tuple[int, int]]:
    for a in range(start, stop, block_size):
        yield a, min(stop, a + block_size)


def _write_split(
    source_root: Path,
    output_root: Path,
    split_name: str,
    start: int,
    stop: int,
    block_size: int,
) -> Path:
    time, lat, lon = _load_coords(source_root)
    split_dir = output_root / f"{split_name}_raw"
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{split_name}.h5"
    n = stop - start
    with h5py.File(out_path, "w") as hf:
        fields = hf.create_dataset(
            "fields",
            shape=(n, len(CHANNELS), len(lat), len(lon)),
            dtype=np.float32,
            chunks=(min(block_size, n), 1, len(lat), len(lon)),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
        )
        hf.create_dataset("channel", data=np.array(CHANNELS, dtype="S"))
        hf.create_dataset("lat", data=lat.astype(np.float32))
        hf.create_dataset("lon", data=lon.astype(np.float32))
        hf.create_dataset("timestamp", data=time[start:stop].astype(np.float64))

        open_arrays = [_open_channel(source_root, ch) for ch in CHANNELS]
        try:
            for a, b in _iter_time_blocks(start, stop, block_size):
                block = np.stack(
                    [da.isel(time=slice(a, b)).values.astype(np.float32) for da in open_arrays],
                    axis=1,
                )
                fields[a - start : b - start] = block
        finally:
            for da in open_arrays:
                da.close()
    return out_path


def _compute_stats(train_h5: Path, output_root: Path, block_size: int) -> Dict[str, str]:
    stats_dir = output_root / "stats_raw"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(train_h5, "r") as hf:
        fields = hf["fields"]
        n_time, n_channel, n_lat, n_lon = fields.shape
        sums = np.zeros((1, n_channel, 1, 1), dtype=np.float64)
        sq_sums = np.zeros((1, n_channel, 1, 1), dtype=np.float64)
        mins = np.full((1, n_channel, 1, 1), np.inf, dtype=np.float32)
        maxs = np.full((1, n_channel, 1, 1), -np.inf, dtype=np.float32)
        time_means = np.zeros((12, n_channel, n_lat, n_lon), dtype=np.float32)
        month_counts = np.zeros(12, dtype=np.int64)

        diff_sums = np.zeros((1, n_channel, 1, 1), dtype=np.float64)
        diff_sq_sums = np.zeros((1, n_channel, 1, 1), dtype=np.float64)
        diff_count = 0
        prev = None
        count = 0

        for a, b in _iter_time_blocks(0, n_time, block_size):
            x = fields[a:b].astype(np.float32)
            valid = np.isfinite(x)
            x_safe = np.where(valid, x, np.nan)
            sums += np.nanmean(x_safe, axis=(0, 2, 3), keepdims=True) * x.shape[0]
            sq_sums += np.nanmean(x_safe * x_safe, axis=(0, 2, 3), keepdims=True) * x.shape[0]
            mins = np.minimum(mins, np.nanmin(x_safe, axis=(0, 2, 3), keepdims=True))
            maxs = np.maximum(maxs, np.nanmax(x_safe, axis=(0, 2, 3), keepdims=True))
            count += x.shape[0]

            for i in range(x.shape[0]):
                m = (a + i) % 12
                time_means[m] += np.nan_to_num(x[i], nan=0.0)
                month_counts[m] += 1

            if prev is not None:
                d0 = x[0:1] - prev
                d = np.concatenate([d0, np.diff(x, axis=0)], axis=0)
            else:
                d = np.diff(x, axis=0)
            if d.shape[0]:
                diff_sums += np.nanmean(d, axis=(0, 2, 3), keepdims=True) * d.shape[0]
                diff_sq_sums += np.nanmean(d * d, axis=(0, 2, 3), keepdims=True) * d.shape[0]
                diff_count += d.shape[0]
            prev = x[-1:]

        global_means = (sums / max(count, 1)).astype(np.float32)
        global_vars = np.maximum((sq_sums / max(count, 1)) - global_means.astype(np.float64) ** 2, 1e-12)
        global_stds = np.sqrt(global_vars).astype(np.float32)

        for m in range(12):
            if month_counts[m] > 0:
                time_means[m] /= float(month_counts[m])

        diff_means = (diff_sums / max(diff_count, 1)).astype(np.float32)
        diff_vars = np.maximum((diff_sq_sums / max(diff_count, 1)) - diff_means.astype(np.float64) ** 2, 1e-12)
        diff_stds = np.sqrt(diff_vars).astype(np.float32)

    paths = {
        "mins": stats_dir / "mins.npy",
        "maxs": stats_dir / "maxs.npy",
        "global_means": stats_dir / "global_means.npy",
        "global_stds": stats_dir / "global_stds.npy",
        "time_means": stats_dir / "time_means.npy",
        "time_diff_means": stats_dir / "time_diff_means.npy",
        "time_diff_stds": stats_dir / "time_diff_stds.npy",
    }
    np.save(paths["mins"], mins)
    np.save(paths["maxs"], maxs)
    np.save(paths["global_means"], global_means)
    np.save(paths["global_stds"], global_stds)
    np.save(paths["time_means"], time_means)
    np.save(paths["time_diff_means"], diff_means)
    np.save(paths["time_diff_stds"], diff_stds)
    return {k: str(v) for k, v in paths.items()}


def _write_metadata(output_root: Path, grid_report: Dict[str, object]) -> Path:
    time, lat, lon = _load_coords(Path(grid_report["source_root"]))
    payload = {
        "dataset_name": "walker_ocean_1deg_full",
        "h5_path": "fields",
        "dhours": 730,
        "coords": {
            "grid_type": "equiangular",
            "channel": CHANNELS,
            "lat": [float(x) for x in lat],
            "lon": [float(x) for x in lon],
        },
        "attrs": {
            "description": "Full 1-degree four-variable WalkerNet ocean/climate fields prepared for Makani.",
            "source_time_units": "days since 0001-01-01 00:00:00, calendar=365_day",
            "source_time_first_last": [float(time[0]), float(time[-1])],
        },
    }
    out = output_root / "metadata" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_root} is not empty; pass --force to reuse it")

    grid_report = _validate_grid(args.source_root)
    grid_report["source_root"] = str(args.source_root)
    splits = _split_indices(int(grid_report["time_length"]))

    split_paths = {}
    for name, (start, stop) in splits.items():
        split_paths[name] = str(_write_split(args.source_root, args.output_root, name, start, stop, args.block_size))

    stats_paths = _compute_stats(Path(split_paths["train"]), args.output_root, args.block_size)
    metadata_path = _write_metadata(args.output_root, grid_report)

    manifest = {
        "dataset": "walker_ocean_1deg_full",
        "channels": CHANNELS,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "split_fractions": SPLITS,
        "split_indices": {k: [int(a), int(b)] for k, (a, b) in splits.items()},
        "split_paths": split_paths,
        "stats_paths": stats_paths,
        "metadata_path": str(metadata_path),
        "grid_report": grid_report,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

