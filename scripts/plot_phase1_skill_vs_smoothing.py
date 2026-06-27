#!/usr/bin/env python3
"""Diagnose long-rollout skill versus smoothing for Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ["tauu", "tauv", "tos", "zos"]
METHODS = ["raw", "fourier", "persistence", "climatology"]
COLORS = {
    "raw": "#7B8794",
    "fourier": "#E76F51",
    "persistence": "#2A9D8F",
    "climatology": "#E9C46A",
}
BANDS = {"low": (1, 5), "mid": (6, 20), "high": (21, 10_000)}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _valid_slots(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        return np.flatnonzero(hf["timestamp"][:] > 0)


def _alignment(raw_forecast: Path, test_h5: Path) -> Tuple[np.ndarray, np.ndarray]:
    slots = _valid_slots(raw_forecast)
    with h5py.File(test_h5, "r") as hf:
        tmap = {float(ts): i for i, ts in enumerate(hf["timestamp"][:])}
    with h5py.File(raw_forecast, "r") as hf:
        base = np.array([tmap[float(ts)] for ts in hf["timestamp"][slots]], dtype=np.int64)
    return slots, base


def _region_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (lat >= -5.0)[:, None] & (lat <= 5.0)[:, None] & (lon >= 190.0)[None, :] & (lon <= 240.0)[None, :]


def _masked_flat(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x[..., mask].reshape(-1)


def _rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    d = _masked_flat(pred - truth, mask)
    return float(np.sqrt(np.nanmean(d * d)))


def _rms(x: np.ndarray, mask: np.ndarray) -> float:
    y = _masked_flat(x, mask)
    return float(np.sqrt(np.nanmean(y * y)))


def _variance(x: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanvar(_masked_flat(x, mask)))


def _acc(pred_anom: np.ndarray, truth_anom: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for p, t in zip(pred_anom, truth_anom):
        x = p[mask].astype(np.float64)
        y = t[mask].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        x = x[ok] - np.mean(x[ok])
        y = y[ok] - np.mean(y[ok])
        den = np.sqrt(np.sum(x * x) * np.sum(y * y))
        if den > 0:
            vals.append(float(np.sum(x * y) / den))
    return float(np.nanmean(vals)) if vals else float("nan")


def _radial_band_power(fields: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    sums = {k: [] for k in BANDS}
    for field in fields:
        x = np.where(mask, field, np.nan).astype(np.float64)
        x = x - np.nanmean(x)
        x = np.nan_to_num(x, nan=0.0)
        power = np.abs(np.fft.rfft2(x)) ** 2
        ky = np.fft.fftfreq(x.shape[0])[:, None] * x.shape[0]
        kx = np.fft.rfftfreq(x.shape[1])[None, :] * x.shape[1]
        kr = np.sqrt(kx * kx + ky * ky)
        for name, (lo, hi) in BANDS.items():
            band = (kr >= lo) & (kr <= hi)
            sums[name].append(float(np.nanmean(power[band])))
    return {k: float(np.nanmean(v)) for k, v in sums.items()}


def _corr_flat(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    a = x[ok] - np.mean(x[ok])
    b = y[ok] - np.mean(y[ok])
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else float("nan")


def _plot_rmse_acc(out: Path, leads: np.ndarray, metrics: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.7), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[0, ci]
        for method in METHODS:
            ax.plot(leads, metrics[f"rmse_{method}"][:, ci], color=COLORS[method], label=method)
        ax.set_title(f"{ch} RMSE")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("RMSE")

        ax = axes[1, ci]
        for method in ["raw", "fourier", "persistence"]:
            ax.plot(leads, metrics[f"acc_{method}"][:, ci], color=COLORS[method], label=method)
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{ch} pattern ACC")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("ACC")
    axes[0, -1].legend(loc="best")
    axes[1, -1].legend(loc="best")
    _save(fig, out / "fig_skill_rmse_acc")


def _plot_smoothing(out: Path, leads: np.ndarray, metrics: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.7), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[0, ci]
        for method in ["raw", "fourier", "persistence"]:
            ax.plot(leads, metrics[f"amp_ratio_{method}"][:, ci], color=COLORS[method], label=method)
        ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{ch} anomaly amplitude")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("Predicted / truth")

        ax = axes[1, ci]
        for method in ["raw", "fourier", "persistence"]:
            ax.plot(leads, metrics[f"var_ratio_{method}"][:, ci], color=COLORS[method], label=method)
        ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{ch} anomaly variance")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("Predicted / truth")
    axes[0, -1].legend(loc="best")
    axes[1, -1].legend(loc="best")
    _save(fig, out / "fig_smoothing_amplitude_variance")


def _plot_spectrum(out: Path, leads: np.ndarray, spectrum: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(10.4, 6.8), constrained_layout=True)
    for ri, band in enumerate(BANDS):
        for ci, ch in enumerate(CHANNELS):
            ax = axes[ri, ci]
            for method in ["raw", "fourier", "persistence"]:
                ax.plot(leads, spectrum[f"{method}_{band}"][:, ci], color=COLORS[method], label=method)
            ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
            ax.set_title(f"{ch} {band}-k energy")
            ax.set_xlabel("Lead months")
            if ci == 0:
                ax.set_ylabel("Predicted / truth")
    axes[0, -1].legend(loc="best")
    _save(fig, out / "fig_spectral_energy_ratios")


def _plot_nino(out: Path, leads: np.ndarray, nino: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2), constrained_layout=True)
    ax = axes[0, 0]
    for method in METHODS:
        ax.plot(leads, nino[f"rmse_{method}"], color=COLORS[method], label=method)
    ax.set_title("Nino3.4 RMSE")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")

    ax = axes[0, 1]
    for method in ["raw", "fourier", "persistence"]:
        ax.plot(leads, nino[f"amp_ratio_{method}"], color=COLORS[method], label=method)
    ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 amplitude ratio")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("Predicted / truth")

    ax = axes[1, 0]
    for method in ["raw", "fourier", "persistence"]:
        ax.plot(leads, nino[f"cum_corr_{method}"], color=COLORS[method], label=method)
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 cumulative correlation")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("Correlation")

    ax = axes[1, 1]
    ax.plot(leads, nino["truth_mean"], color="#264653", label="truth")
    ax.plot(leads, nino["raw_mean"], color=COLORS["raw"], label="raw")
    ax.plot(leads, nino["fourier_mean"], color=COLORS["fourier"], label="fourier")
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 lead-mean anomaly")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("Anomaly")
    _save(fig, out / "fig_nino34_skill_vs_smoothing")


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
    manifest = json.load(open(root / "data" / "walker_ocean_1deg_full" / "manifest.json", encoding="utf-8"))
    test_global_start = int(manifest["split_indices"]["test"][0])
    time_means = np.load(root / "data" / "walker_ocean_1deg_full" / "stats_raw" / "time_means.npy")
    valid_mask = np.load(root / "data" / "walker_ocean_1deg_full" / "stats_raw" / "valid_mask.npy").astype(bool)
    out = args.output_dir or (root / "figures" / f"phase1_skill_vs_smoothing_rollout{args.rollout_months}")
    out.mkdir(parents=True, exist_ok=True)

    slots, base = _alignment(raw_forecast, test_h5)
    leads = np.arange(args.rollout_months + 1)
    n_leads = len(leads)
    n_ch = len(CHANNELS)

    metric_names = []
    for prefix in ["rmse", "acc", "amp_ratio", "var_ratio"]:
        methods = METHODS if prefix == "rmse" else ["raw", "fourier", "persistence"]
        metric_names += [f"{prefix}_{m}" for m in methods]
    metrics = {name: np.full((n_leads, n_ch), np.nan, dtype=np.float64) for name in metric_names}
    spectrum = {f"{m}_{b}": np.full((n_leads, n_ch), np.nan, dtype=np.float64) for m in ["raw", "fourier", "persistence"] for b in BANDS}
    nino = {f"rmse_{m}": np.full(n_leads, np.nan, dtype=np.float64) for m in METHODS}
    nino.update({f"amp_ratio_{m}": np.full(n_leads, np.nan, dtype=np.float64) for m in ["raw", "fourier", "persistence"]})
    nino.update({f"cum_corr_{m}": np.full(n_leads, np.nan, dtype=np.float64) for m in ["raw", "fourier", "persistence"]})
    nino.update({k: np.full(n_leads, np.nan, dtype=np.float64) for k in ["truth_mean", "raw_mean", "fourier_mean"]})
    nino_traj = {m: np.full((len(slots), n_leads), np.nan, dtype=np.float64) for m in ["truth", "raw", "fourier", "persistence", "climatology"]}

    with h5py.File(test_h5, "r") as ht, h5py.File(raw_forecast, "r") as hr, h5py.File(fourier_forecast, "r") as hf:
        lat = ht["lat"][:]
        lon = ht["lon"][:]
        nino_mask = _region_mask(lat, lon) & valid_mask[2]
        initial = ht["fields"][base, :, :, :].astype(np.float32)
        for lead in leads:
            target = base + int(lead)
            months = (test_global_start + target) % 12
            truth = ht["fields"][target, :, :, :].astype(np.float32)
            raw = hr["fields"][slots, int(lead), 0, :, :, :].astype(np.float32)
            fourier = hf["fields"][slots, int(lead), 0, :, :, :].astype(np.float32)
            climatology = time_means[months].astype(np.float32)
            fields = {"raw": raw, "fourier": fourier, "persistence": initial, "climatology": climatology}

            truth_anom = truth - climatology
            anoms = {m: x - climatology for m, x in fields.items()}
            for ci in range(n_ch):
                mask = valid_mask[ci]
                truth_rms = max(_rms(truth_anom[:, ci], mask), 1e-12)
                truth_var = max(_variance(truth_anom[:, ci], mask), 1e-12)
                truth_bands = _radial_band_power(truth_anom[:, ci], mask)
                for method, field in fields.items():
                    metrics[f"rmse_{method}"][lead, ci] = _rmse(field[:, ci], truth[:, ci], mask)
                for method in ["raw", "fourier", "persistence"]:
                    metrics[f"acc_{method}"][lead, ci] = _acc(anoms[method][:, ci], truth_anom[:, ci], mask)
                    metrics[f"amp_ratio_{method}"][lead, ci] = _rms(anoms[method][:, ci], mask) / truth_rms
                    metrics[f"var_ratio_{method}"][lead, ci] = _variance(anoms[method][:, ci], mask) / truth_var
                    band_power = _radial_band_power(anoms[method][:, ci], mask)
                    for band in BANDS:
                        spectrum[f"{method}_{band}"][lead, ci] = band_power[band] / max(truth_bands[band], 1e-30)

            clim_nino = np.array([np.nanmean(np.where(nino_mask, time_means[m, 2], np.nan)) for m in months])
            nino_traj["truth"][:, lead] = np.nanmean(np.where(nino_mask, truth[:, 2], np.nan), axis=(1, 2)) - clim_nino
            for method, field in fields.items():
                nino_traj[method][:, lead] = np.nanmean(np.where(nino_mask, field[:, 2], np.nan), axis=(1, 2)) - clim_nino
                nino[f"rmse_{method}"][lead] = float(np.sqrt(np.nanmean((nino_traj[method][:, lead] - nino_traj["truth"][:, lead]) ** 2)))
            truth_nino_rms = max(float(np.sqrt(np.nanmean(nino_traj["truth"][:, lead] ** 2))), 1e-12)
            for method in ["raw", "fourier", "persistence"]:
                nino[f"amp_ratio_{method}"][lead] = float(np.sqrt(np.nanmean(nino_traj[method][:, lead] ** 2)) / truth_nino_rms)
                nino[f"cum_corr_{method}"][lead] = _corr_flat(nino_traj[method][:, : lead + 1].ravel(), nino_traj["truth"][:, : lead + 1].ravel())
            nino["truth_mean"][lead] = float(np.nanmean(nino_traj["truth"][:, lead]))
            nino["raw_mean"][lead] = float(np.nanmean(nino_traj["raw"][:, lead]))
            nino["fourier_mean"][lead] = float(np.nanmean(nino_traj["fourier"][:, lead]))

    _plot_rmse_acc(out, leads, metrics)
    _plot_smoothing(out, leads, metrics)
    _plot_spectrum(out, leads, spectrum)
    _plot_nino(out, leads, nino)

    summary = {
        "valid_rollout_slots": slots.astype(int).tolist(),
        "base_test_indices": base.astype(int).tolist(),
        "channels": CHANNELS,
        "bands": BANDS,
        "metrics": {k: v.tolist() for k, v in metrics.items()},
        "spectrum": {k: v.tolist() for k, v in spectrum.items()},
        "nino34": {k: v.tolist() for k, v in nino.items()},
    }
    (out / "skill_vs_smoothing_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "skill_vs_smoothing_diagnostics.npz",
        leads=leads,
        **metrics,
        **{f"spectrum_{k}": v for k, v in spectrum.items()},
        **{f"nino34_{k}": v for k, v in nino.items()},
    )
    print(json.dumps({"output_dir": str(out), "png_count": len(list(out.glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
