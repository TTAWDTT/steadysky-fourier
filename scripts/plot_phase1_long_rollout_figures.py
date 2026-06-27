#!/usr/bin/env python3
"""Plot Phase-1 long-rollout diagnostics.

The script reads Makani forecast HDF5 files plus the held-out test split and
produces paper-style figures for Nino3.4, spatial fields, spatial errors,
radial spectra, and global drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ["tauu", "tauv", "tos", "zos"]
SELECTED_LEADS = [1, 3, 6, 12, 30, 60, 120]
RAW_COLOR = "#7B8794"
FOURIER_COLOR = "#E76F51"
TRUTH_COLOR = "#264653"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 1.8,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _decode_channels(hf: h5py.File) -> List[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in hf["channel"][:]]


def _valid_forecast_slots(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        timestamps = hf["timestamp"][:]
    return np.flatnonzero(timestamps > 0)


def _test_timestamp_index(test_h5: Path) -> Dict[float, int]:
    with h5py.File(test_h5, "r") as hf:
        timestamps = hf["timestamp"][:]
    return {float(ts): int(i) for i, ts in enumerate(timestamps)}


def _rollout_alignment(raw_forecast: Path, test_h5: Path) -> Tuple[np.ndarray, np.ndarray]:
    timestamp_to_index = _test_timestamp_index(test_h5)
    slots = _valid_forecast_slots(raw_forecast)
    with h5py.File(raw_forecast, "r") as hf:
        timestamps = hf["timestamp"][slots]
    base_indices = np.array([timestamp_to_index[float(ts)] for ts in timestamps], dtype=np.int64)
    return slots, base_indices


def _masked_mean(x: np.ndarray, mask: np.ndarray, axes=None) -> np.ndarray:
    m = mask.astype(bool)
    while m.ndim < x.ndim:
        m = np.expand_dims(m, axis=0)
    y = np.where(m, x, np.nan)
    return np.nanmean(y, axis=axes)


def _region_mask(lat: np.ndarray, lon: np.ndarray, lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float]) -> np.ndarray:
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    return lat_mask[:, None] & lon_mask[None, :]


def _robust_limits(arrays: Iterable[np.ndarray], symmetric: bool = False) -> Tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(vals), 98))
        return -vmax, vmax
    return float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))


def _radial_spectrum(fields: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return average isotropic 2D power spectrum for fields shaped (n, lat, lon)."""
    spectra = []
    for field in fields:
        x = np.where(mask, field, np.nan).astype(np.float64)
        x = x - np.nanmean(x)
        x = np.nan_to_num(x, nan=0.0)
        power = np.abs(np.fft.rfft2(x)) ** 2
        ky = np.fft.fftfreq(x.shape[0])[:, None] * x.shape[0]
        kx = np.fft.rfftfreq(x.shape[1])[None, :] * x.shape[1]
        kr = np.sqrt(kx * kx + ky * ky).astype(np.int64)
        max_k = int(kr.max())
        radial = np.bincount(kr.ravel(), weights=power.ravel(), minlength=max_k + 1)
        counts = np.bincount(kr.ravel(), minlength=max_k + 1)
        spectra.append(radial / np.maximum(counts, 1))
    spec = np.mean(np.stack(spectra, axis=0), axis=0)
    return np.arange(spec.shape[0]), spec


def _read_forecast(path: Path, slots: np.ndarray, lead: int) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        return hf["fields"][slots, lead, 0, :, :, :].astype(np.float32)


def _read_truth(test_h5: Path, target_indices: np.ndarray) -> np.ndarray:
    with h5py.File(test_h5, "r") as hf:
        return hf["fields"][target_indices, :, :, :].astype(np.float32)


def plot_nino34(
    out_dir: Path,
    lead_months: np.ndarray,
    test_months: np.ndarray,
    truth_full: np.ndarray,
    base_indices: np.ndarray,
    raw_traj: np.ndarray,
    fourier_traj: np.ndarray,
    raw_rmse: np.ndarray,
    fourier_rmse: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), constrained_layout=True)
    ax = axes[0]
    ax.plot(test_months, truth_full, color=TRUTH_COLOR, label="Truth", linewidth=2.0)
    for i, base in enumerate(base_indices):
        label_raw = "Raw rollout" if i == 0 else None
        label_fourier = "Fourier rollout" if i == 0 else None
        x = base + lead_months
        ax.plot(x, raw_traj[i], color=RAW_COLOR, alpha=0.45, linewidth=1.0, label=label_raw)
        ax.plot(x, fourier_traj[i], color=FOURIER_COLOR, alpha=0.55, linewidth=1.0, label=label_fourier)
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 tos anomaly trajectories")
    ax.set_xlabel("Test month index")
    ax.set_ylabel("Anomaly")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(lead_months, raw_rmse, color=RAW_COLOR, label="Raw baseline")
    ax.plot(lead_months, fourier_rmse, color=FOURIER_COLOR, label="Fourier curriculum")
    ax.set_title("Nino3.4 rollout error")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")
    _save(fig, out_dir / "fig_nino34_curve")


def plot_nino34_clean(
    out_dir: Path,
    lead_months: np.ndarray,
    base_indices: np.ndarray,
    truth_traj: np.ndarray,
    raw_traj: np.ndarray,
    fourier_traj: np.ndarray,
    raw_rmse: np.ndarray,
    fourier_rmse: np.ndarray,
) -> None:
    truth_mean = truth_traj.mean(axis=0)
    raw_mean = raw_traj.mean(axis=0)
    fourier_mean = fourier_traj.mean(axis=0)
    truth_lo, truth_hi = np.percentile(truth_traj, [10, 90], axis=0)
    raw_lo, raw_hi = np.percentile(raw_traj, [10, 90], axis=0)
    fourier_lo, fourier_hi = np.percentile(fourier_traj, [10, 90], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), constrained_layout=True)
    ax = axes[0]
    for mean, lo, hi, color, label in [
        (truth_mean, truth_lo, truth_hi, TRUTH_COLOR, "Truth"),
        (raw_mean, raw_lo, raw_hi, RAW_COLOR, "Raw baseline"),
        (fourier_mean, fourier_lo, fourier_hi, FOURIER_COLOR, "Fourier curriculum"),
    ]:
        ax.plot(lead_months, mean, color=color, label=label)
        ax.fill_between(lead_months, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 anomaly mean and 10-90% range")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("Anomaly")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(lead_months, raw_rmse, color=RAW_COLOR, label="Raw baseline")
    ax.plot(lead_months, fourier_rmse, color=FOURIER_COLOR, label="Fourier curriculum")
    ax.set_title("Nino3.4 rollout RMSE")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")
    _save(fig, out_dir / "fig_nino34_summary")

    fig, axes = plt.subplots(4, 2, figsize=(8.0, 8.2), constrained_layout=True)
    axes = axes.flat
    for i, base in enumerate(base_indices):
        ax = axes[i]
        ax.plot(lead_months, truth_traj[i], color=TRUTH_COLOR, label="Truth")
        ax.plot(lead_months, raw_traj[i], color=RAW_COLOR, label="Raw")
        ax.plot(lead_months, fourier_traj[i], color=FOURIER_COLOR, label="Fourier")
        ax.axhline(0.0, color="#333333", linewidth=0.7, alpha=0.35)
        ax.set_title(f"IC test month {int(base)}")
        ax.set_xlabel("Lead")
        ax.set_ylabel("Anom.")
    axes[-1].axis("off")
    axes[0].legend(loc="best", ncol=3)
    _save(fig, out_dir / "fig_nino34_by_initial_condition")


def plot_nino34_lead_mean(
    out_dir: Path,
    lead_months: np.ndarray,
    truth: np.ndarray,
    raw: np.ndarray,
    fourier: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 2.7), constrained_layout=True)
    ax.plot(lead_months, truth, color=TRUTH_COLOR, label="Truth lead-mean")
    ax.plot(lead_months, raw, color=RAW_COLOR, label="Raw lead-mean")
    ax.plot(lead_months, fourier, color=FOURIER_COLOR, label="Fourier lead-mean")
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 lead-mean diagnostic")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("Anomaly averaged over valid ICs")
    ax.legend(loc="best")
    _save(fig, out_dir / "fig_nino34_lead_mean_diagnostic")


def plot_drift(out_dir: Path, lead_months: np.ndarray, drift: Dict[str, np.ndarray], rmse: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10.2, 4.6), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[0, ci]
        ax.plot(lead_months, drift["raw"][:, ci], color=RAW_COLOR, label="Raw")
        ax.plot(lead_months, drift["fourier"][:, ci], color=FOURIER_COLOR, label="Fourier")
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{ch} drift")
        ax.set_xlabel("Lead")
        if ci == 0:
            ax.set_ylabel("Mean forecast - truth")

        ax = axes[1, ci]
        ax.plot(lead_months, rmse["raw"][:, ci], color=RAW_COLOR, label="Raw")
        ax.plot(lead_months, rmse["fourier"][:, ci], color=FOURIER_COLOR, label="Fourier")
        ax.set_title(f"{ch} RMSE")
        ax.set_xlabel("Lead")
        if ci == 0:
            ax.set_ylabel("RMSE")
    axes[0, -1].legend(loc="best")
    axes[1, -1].legend(loc="best")
    _save(fig, out_dir / "fig_drift_rmse_curves")


def plot_spatial(out_dir: Path, lead: int, lat: np.ndarray, lon: np.ndarray, truth: np.ndarray, raw: np.ndarray, fourier: np.ndarray) -> None:
    truth_mean = truth.mean(axis=0)
    raw_mean = raw.mean(axis=0)
    fourier_mean = fourier.mean(axis=0)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    fig, axes = plt.subplots(4, 3, figsize=(9.2, 7.6), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        vmin, vmax = _robust_limits([truth_mean[ci], raw_mean[ci], fourier_mean[ci]])
        for j, (name, arr) in enumerate([("Truth", truth_mean), ("Raw", raw_mean), ("Fourier", fourier_mean)]):
            ax = axes[ci, j]
            im = ax.imshow(arr[ci], origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{name} {ch}")
            ax.set_xlabel("Lon")
            if j == 0:
                ax.set_ylabel("Lat")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"Spatial fields at lead {lead} months", fontweight="bold")
    _save(fig, out_dir / f"fig_spatial_lead{lead:03d}")


def plot_error(out_dir: Path, lead: int, lat: np.ndarray, lon: np.ndarray, truth: np.ndarray, raw: np.ndarray, fourier: np.ndarray) -> None:
    raw_err = (raw - truth).mean(axis=0)
    fourier_err = (fourier - truth).mean(axis=0)
    improvement = np.abs(raw - truth).mean(axis=0) - np.abs(fourier - truth).mean(axis=0)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    fig, axes = plt.subplots(4, 3, figsize=(9.2, 7.6), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        evmin, evmax = _robust_limits([raw_err[ci], fourier_err[ci]], symmetric=True)
        ivmin, ivmax = _robust_limits([improvement[ci]], symmetric=True)
        panels = [
            ("Raw - truth", raw_err[ci], "coolwarm", evmin, evmax),
            ("Fourier - truth", fourier_err[ci], "coolwarm", evmin, evmax),
            ("|Raw err| - |Fourier err|", improvement[ci], "RdBu_r", ivmin, ivmax),
        ]
        for j, (title, arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[ci, j]
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{title} {ch}")
            ax.set_xlabel("Lon")
            if j == 0:
                ax.set_ylabel("Lat")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"Spatial errors at lead {lead} months", fontweight="bold")
    _save(fig, out_dir / f"fig_error_lead{lead:03d}")


def plot_spectrum(out_dir: Path, lead: int, truth: np.ndarray, raw: np.ndarray, fourier: np.ndarray, valid_mask: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes.flat[ci]
        mask = valid_mask[ci].astype(bool)
        for name, arr, color in [
            ("Truth", truth[:, ci], TRUTH_COLOR),
            ("Raw", raw[:, ci], RAW_COLOR),
            ("Fourier", fourier[:, ci], FOURIER_COLOR),
        ]:
            k, spec = _radial_spectrum(arr, mask)
            ax.loglog(k[1:], spec[1:] + 1e-30, color=color, label=name)
        ax.set_title(ch)
        ax.set_xlabel("Radial wavenumber")
        ax.set_ylabel("Power")
    axes.flat[0].legend(loc="best")
    fig.suptitle(f"Radial spectra at lead {lead} months", fontweight="bold")
    _save(fig, out_dir / f"fig_spectrum_lead{lead:03d}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=Path("/mnt/nvme1/lz/fourier_layerwise_weather"))
    parser.add_argument("--rollout-months", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    _style()
    root = args.work_root
    run_root = root / "runs" / "sfno_walker_1deg_edim384_layers8"
    raw_forecast = run_root / "phase1_raw_edim384" / "scores" / f"phase1_raw_rollout{args.rollout_months}_forecasts.h5"
    fourier_forecast = run_root / "phase1_fourier_edim384" / "scores" / f"phase1_fourier_rollout{args.rollout_months}_forecasts.h5"
    test_h5 = root / "data" / "walker_ocean_1deg_full" / "test_raw" / "test.h5"
    manifest_path = root / "data" / "walker_ocean_1deg_full" / "manifest.json"
    time_means_path = root / "data" / "walker_ocean_1deg_full" / "stats_raw" / "time_means.npy"
    valid_mask_path = root / "data" / "walker_ocean_1deg_full" / "stats_raw" / "valid_mask.npy"
    out_dir = args.output_dir or (root / "figures" / f"phase1_rollout{args.rollout_months}")
    out_dir.mkdir(parents=True, exist_ok=True)

    slots, base_indices = _rollout_alignment(raw_forecast, test_h5)
    with h5py.File(test_h5, "r") as hf:
        lat = hf["lat"][:].astype(np.float32)
        lon = hf["lon"][:].astype(np.float32)
        channels = _decode_channels(hf)
    if channels != CHANNELS:
        raise ValueError(f"Unexpected channel order: {channels}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    test_global_start = int(manifest["split_indices"]["test"][0])
    time_means = np.load(time_means_path)
    valid_mask = np.load(valid_mask_path).astype(bool)

    lead_months = np.arange(args.rollout_months + 1)
    n_leads = len(lead_months)
    nino_mask = _region_mask(lat, lon, (-5.0, 5.0), (190.0, 240.0)) & valid_mask[2]

    nino = {k: np.zeros(n_leads, dtype=np.float64) for k in ["truth", "raw", "fourier"]}
    nino_traj = {k: np.zeros((len(slots), n_leads), dtype=np.float64) for k in ["truth", "raw", "fourier"]}
    nino_rmse = {k: np.zeros(n_leads, dtype=np.float64) for k in ["raw", "fourier"]}
    drift = {k: np.zeros((n_leads, len(CHANNELS)), dtype=np.float64) for k in ["raw", "fourier"]}
    rmse = {k: np.zeros((n_leads, len(CHANNELS)), dtype=np.float64) for k in ["raw", "fourier"]}

    with h5py.File(test_h5, "r") as hf:
        test_tos = hf["fields"][:, 2, :, :].astype(np.float32)
    test_month_ids = (test_global_start + np.arange(test_tos.shape[0])) % 12
    test_clim_nino = np.array([np.nanmean(np.where(nino_mask, time_means[m, 2], np.nan)) for m in test_month_ids])
    truth_nino_full = _masked_mean(test_tos, nino_mask, axes=(1, 2)) - test_clim_nino

    selected_payload = {}
    for lead in lead_months:
        target_indices = base_indices + int(lead)
        truth = _read_truth(test_h5, target_indices)
        raw = _read_forecast(raw_forecast, slots, int(lead))
        fourier = _read_forecast(fourier_forecast, slots, int(lead))

        target_months = (test_global_start + target_indices) % 12
        clim_nino = np.array([np.nanmean(np.where(nino_mask, time_means[m, 2], np.nan)) for m in target_months])
        truth_nino_ic = _masked_mean(truth[:, 2], nino_mask, axes=(1, 2)) - clim_nino
        raw_nino_ic = _masked_mean(raw[:, 2], nino_mask, axes=(1, 2)) - clim_nino
        fourier_nino_ic = _masked_mean(fourier[:, 2], nino_mask, axes=(1, 2)) - clim_nino
        nino_traj["truth"][:, lead] = truth_nino_ic
        nino_traj["raw"][:, lead] = raw_nino_ic
        nino_traj["fourier"][:, lead] = fourier_nino_ic
        nino["truth"][lead] = float(np.nanmean(truth_nino_ic))
        nino["raw"][lead] = float(np.nanmean(raw_nino_ic))
        nino["fourier"][lead] = float(np.nanmean(fourier_nino_ic))
        nino_rmse["raw"][lead] = float(np.sqrt(np.nanmean((raw_nino_ic - truth_nino_ic) ** 2)))
        nino_rmse["fourier"][lead] = float(np.sqrt(np.nanmean((fourier_nino_ic - truth_nino_ic) ** 2)))

        for ci in range(len(CHANNELS)):
            mask = valid_mask[ci]
            raw_err = raw[:, ci] - truth[:, ci]
            fourier_err = fourier[:, ci] - truth[:, ci]
            drift["raw"][lead, ci] = float(_masked_mean(raw_err, mask))
            drift["fourier"][lead, ci] = float(_masked_mean(fourier_err, mask))
            rmse["raw"][lead, ci] = float(np.sqrt(_masked_mean(raw_err * raw_err, mask)))
            rmse["fourier"][lead, ci] = float(np.sqrt(_masked_mean(fourier_err * fourier_err, mask)))

        if int(lead) in SELECTED_LEADS:
            selected_payload[int(lead)] = (truth, raw, fourier)

    plot_nino34(
        out_dir,
        lead_months,
        np.arange(truth_nino_full.shape[0]),
        truth_nino_full,
        base_indices,
        nino_traj["raw"],
        nino_traj["fourier"],
        nino_rmse["raw"],
        nino_rmse["fourier"],
    )
    plot_nino34_clean(
        out_dir,
        lead_months,
        base_indices,
        nino_traj["truth"],
        nino_traj["raw"],
        nino_traj["fourier"],
        nino_rmse["raw"],
        nino_rmse["fourier"],
    )
    plot_nino34_lead_mean(out_dir, lead_months, nino["truth"], nino["raw"], nino["fourier"])
    plot_drift(out_dir, lead_months, drift, rmse)

    for lead in SELECTED_LEADS:
        truth, raw, fourier = selected_payload[lead]
        plot_spatial(out_dir, lead, lat, lon, truth, raw, fourier)
        plot_error(out_dir, lead, lat, lon, truth, raw, fourier)
        plot_spectrum(out_dir, lead, truth, raw, fourier, valid_mask)

    summary = {
        "valid_rollout_slots": slots.astype(int).tolist(),
        "base_test_indices": base_indices.astype(int).tolist(),
        "selected_leads_months": SELECTED_LEADS,
        "nino34": {k: v.tolist() for k, v in nino.items()},
        "nino34_truth_full_test": truth_nino_full.tolist(),
        "nino34_trajectories": {k: v.tolist() for k, v in nino_traj.items()},
        "nino34_rmse": {k: v.tolist() for k, v in nino_rmse.items()},
        "nino34_lead0_max_abs_error": {
            "raw": float(np.max(np.abs(nino_traj["raw"][:, 0] - nino_traj["truth"][:, 0]))),
            "fourier": float(np.max(np.abs(nino_traj["fourier"][:, 0] - nino_traj["truth"][:, 0]))),
        },
        "drift": {k: v.tolist() for k, v in drift.items()},
        "rmse": {k: v.tolist() for k, v in rmse.items()},
        "channels": CHANNELS,
        "nino34_region": {"lat": [-5.0, 5.0], "lon": [190.0, 240.0], "variable": "tos"},
    }
    (out_dir / "phase1_rollout_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "phase1_rollout_diagnostics.npz",
        lead_months=lead_months,
        nino_truth=nino["truth"],
        nino_truth_full_test=truth_nino_full,
        nino_traj_truth=nino_traj["truth"],
        nino_traj_raw=nino_traj["raw"],
        nino_traj_fourier=nino_traj["fourier"],
        nino_raw=nino["raw"],
        nino_fourier=nino["fourier"],
        nino_rmse_raw=nino_rmse["raw"],
        nino_rmse_fourier=nino_rmse["fourier"],
        drift_raw=drift["raw"],
        drift_fourier=drift["fourier"],
        rmse_raw=rmse["raw"],
        rmse_fourier=rmse["fourier"],
    )
    print(json.dumps({"output_dir": str(out_dir), "figure_count_png": len(list(out_dir.glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
