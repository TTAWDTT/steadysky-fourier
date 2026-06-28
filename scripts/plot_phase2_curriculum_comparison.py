#!/usr/bin/env python3
"""Compare Phase-2 curriculum variants against Phase-1 baselines.

The diagnostics are deliberately focused on the central question raised after
Phase 1: do curriculum variants improve long-rollout skill without merely
shrinking anomalies toward a smoother attractor?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ["tauu", "tauv", "tos", "zos"]
MODEL_METHODS = ["raw", "fourier", "mixed", "residual"]
REFERENCE_METHODS = ["persistence", "climatology"]
ALL_METHODS = MODEL_METHODS + REFERENCE_METHODS
SKILL_METHODS = MODEL_METHODS + ["persistence"]
BANDS = {"low": (1, 5), "mid": (6, 20), "high": (21, 10_000)}
COLORS = {
    "raw": "#667085",
    "fourier": "#D1495B",
    "mixed": "#0077B6",
    "residual": "#7A4CC2",
    "persistence": "#2A9D8F",
    "climatology": "#E9C46A",
    "truth": "#264653",
}
LABELS = {
    "raw": "Raw",
    "fourier": "Pure Fourier",
    "mixed": "Mixed",
    "residual": "Residual",
    "persistence": "Persistence",
    "climatology": "Climatology",
    "truth": "Truth",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    forecast: Path


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.4,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.55,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _decode_channels(hf: h5py.File) -> List[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in hf["channel"][:]]


def _valid_slots(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        return np.flatnonzero(hf["timestamp"][:] > 0)


def _alignment(reference_forecast: Path, test_h5: Path) -> Tuple[np.ndarray, np.ndarray]:
    slots = _valid_slots(reference_forecast)
    with h5py.File(test_h5, "r") as hf:
        timestamp_to_index = {float(ts): int(i) for i, ts in enumerate(hf["timestamp"][:])}
    with h5py.File(reference_forecast, "r") as hf:
        base = np.array([timestamp_to_index[float(ts)] for ts in hf["timestamp"][slots]], dtype=np.int64)
    return slots, base


def _region_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (lat >= -5.0)[:, None] & (lat <= 5.0)[:, None] & (lon >= 190.0)[None, :] & (lon <= 240.0)[None, :]


def _masked_flat(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x[..., mask].reshape(-1)


def _masked_mean(x: np.ndarray, mask: np.ndarray, axes=None) -> np.ndarray:
    m = mask.astype(bool)
    while m.ndim < x.ndim:
        m = np.expand_dims(m, axis=0)
    return np.nanmean(np.where(m, x, np.nan), axis=axes)


def _rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    d = _masked_flat(pred - truth, mask)
    return float(np.sqrt(np.nanmean(d * d)))


def _rms(x: np.ndarray, mask: np.ndarray) -> float:
    y = _masked_flat(x, mask)
    return float(np.sqrt(np.nanmean(y * y)))


def _variance(x: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanvar(_masked_flat(x, mask)))


def _corr_flat(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    a = x[ok] - np.mean(x[ok])
    b = y[ok] - np.mean(y[ok])
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else float("nan")


def _acc(pred_anom: np.ndarray, truth_anom: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for pred, truth in zip(pred_anom, truth_anom):
        vals.append(_corr_flat(pred[mask].astype(np.float64), truth[mask].astype(np.float64)))
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


def _forecast_paths(run_root: Path, rollout_months: int) -> List[MethodSpec]:
    return [
        MethodSpec("raw", run_root / "phase1_raw_edim384" / "scores" / f"phase1_raw_rollout{rollout_months}_forecasts.h5"),
        MethodSpec("fourier", run_root / "phase1_fourier_edim384" / "scores" / f"phase1_fourier_rollout{rollout_months}_forecasts.h5"),
        MethodSpec("mixed", run_root / "phase2_mixed_edim384" / "scores" / f"phase2_mixed_edim384_rollout{rollout_months}_forecasts.h5"),
        MethodSpec("residual", run_root / "phase2_residual_edim384" / "scores" / f"phase2_residual_edim384_rollout{rollout_months}_forecasts.h5"),
    ]


def _read_forecasts(paths: Dict[str, h5py.File], slots: np.ndarray, lead: int) -> Dict[str, np.ndarray]:
    return {name: hf["fields"][slots, lead, 0, :, :, :].astype(np.float32) for name, hf in paths.items()}


def _plot_rmse_acc(out: Path, leads: np.ndarray, metrics: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 4.9), constrained_layout=True)
    for ci, channel in enumerate(CHANNELS):
        ax = axes[0, ci]
        for method in ALL_METHODS:
            ax.plot(leads, metrics[f"rmse_{method}"][:, ci], color=COLORS[method], label=LABELS[method])
        ax.set_title(f"{channel} RMSE")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("RMSE")

        ax = axes[1, ci]
        for method in SKILL_METHODS:
            ax.plot(leads, metrics[f"acc_{method}"][:, ci], color=COLORS[method], label=LABELS[method])
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{channel} pattern ACC")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("ACC")
    axes[0, -1].legend(loc="best")
    axes[1, -1].legend(loc="best")
    _save(fig, out / "fig_phase2_skill_rmse_acc")


def _plot_smoothing(out: Path, leads: np.ndarray, metrics: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 4.9), constrained_layout=True)
    for ci, channel in enumerate(CHANNELS):
        ax = axes[0, ci]
        for method in SKILL_METHODS:
            ax.plot(leads, metrics[f"amp_ratio_{method}"][:, ci], color=COLORS[method], label=LABELS[method])
        ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{channel} anomaly amplitude")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("Predicted / truth")

        ax = axes[1, ci]
        for method in SKILL_METHODS:
            ax.plot(leads, metrics[f"var_ratio_{method}"][:, ci], color=COLORS[method], label=LABELS[method])
        ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{channel} anomaly variance")
        ax.set_xlabel("Lead months")
        if ci == 0:
            ax.set_ylabel("Predicted / truth")
    axes[0, -1].legend(loc="best")
    axes[1, -1].legend(loc="best")
    _save(fig, out / "fig_phase2_smoothing_amplitude_variance")


def _plot_spectrum(out: Path, leads: np.ndarray, spectrum: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(11.2, 7.1), constrained_layout=True)
    for ri, band in enumerate(BANDS):
        for ci, channel in enumerate(CHANNELS):
            ax = axes[ri, ci]
            for method in SKILL_METHODS:
                ax.plot(leads, spectrum[f"{method}_{band}"][:, ci], color=COLORS[method], label=LABELS[method])
            ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
            ax.set_title(f"{channel} {band}-k energy")
            ax.set_xlabel("Lead months")
            if ci == 0:
                ax.set_ylabel("Predicted / truth")
    axes[0, -1].legend(loc="best")
    _save(fig, out / "fig_phase2_spectral_energy_ratios")


def _plot_nino(out: Path, leads: np.ndarray, nino: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.5), constrained_layout=True)
    ax = axes[0, 0]
    for method in ALL_METHODS:
        ax.plot(leads, nino[f"rmse_{method}"], color=COLORS[method], label=LABELS[method])
    ax.set_title("Nino3.4 RMSE")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best", ncol=2)

    ax = axes[0, 1]
    for method in SKILL_METHODS:
        ax.plot(leads, nino[f"amp_ratio_{method}"], color=COLORS[method], label=LABELS[method])
    ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 amplitude ratio")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("Predicted / truth")

    ax = axes[1, 0]
    for method in SKILL_METHODS:
        ax.plot(leads, nino[f"cum_corr_{method}"], color=COLORS[method], label=LABELS[method])
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 cumulative correlation")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("Correlation")

    ax = axes[1, 1]
    ax.plot(leads, nino["truth_mean"], color=COLORS["truth"], label=LABELS["truth"])
    for method in MODEL_METHODS:
        ax.plot(leads, nino[f"{method}_mean"], color=COLORS[method], label=LABELS[method])
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.4)
    ax.set_title("Nino3.4 lead-mean anomaly")
    ax.set_xlabel("Lead months")
    ax.set_ylabel("Anomaly")
    ax.legend(loc="best", ncol=2)
    _save(fig, out / "fig_phase2_nino34_skill_vs_smoothing")


def _plot_summary_bars(out: Path, metrics: Dict[str, np.ndarray], nino: Dict[str, np.ndarray], rollout_months: int) -> None:
    leads = [12, 60, rollout_months]
    labels = [f"{lead}m" for lead in leads]
    x = np.arange(len(leads))
    width = 0.16

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), constrained_layout=True)
    panels = [
        ("tos RMSE", {m: metrics[f"rmse_{m}"][leads, 2] for m in MODEL_METHODS}),
        ("tos ACC", {m: metrics[f"acc_{m}"][leads, 2] for m in MODEL_METHODS}),
        ("Nino3.4 amplitude", {m: nino[f"amp_ratio_{m}"][leads] for m in MODEL_METHODS}),
    ]
    for ax, (title, values) in zip(axes, panels):
        for i, method in enumerate(MODEL_METHODS):
            ax.bar(x + (i - 1.5) * width, values[method], width=width, color=COLORS[method], label=LABELS[method])
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_xlabel("Lead")
        if "amplitude" in title:
            ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.4)
    axes[-1].legend(loc="best")
    _save(fig, out / "fig_phase2_key_lead_summary")


def _copy_key_pngs(out: Path, asset_dir: Path | None) -> None:
    if asset_dir is None:
        return
    asset_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(out.glob("fig_phase2_*.png")):
        target = asset_dir / png.name
        target.write_bytes(png.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=Path("/mnt/nvme1/lz/fourier_layerwise_weather"))
    parser.add_argument("--rollout-months", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--asset-dir", type=Path, default=None)
    args = parser.parse_args()

    _style()
    root = args.work_root
    run_root = root / "runs" / "sfno_walker_1deg_edim384_layers8"
    forecasts = _forecast_paths(run_root, args.rollout_months)
    for spec in forecasts:
        if not spec.forecast.exists():
            raise FileNotFoundError(f"Missing forecast for {spec.name}: {spec.forecast}")

    test_h5 = root / "data" / "walker_ocean_1deg_full" / "test_raw" / "test.h5"
    manifest = json.loads((root / "data" / "walker_ocean_1deg_full" / "manifest.json").read_text(encoding="utf-8"))
    test_global_start = int(manifest["split_indices"]["test"][0])
    time_means = np.load(root / "data" / "walker_ocean_1deg_full" / "stats_raw" / "time_means.npy")
    valid_mask = np.load(root / "data" / "walker_ocean_1deg_full" / "stats_raw" / "valid_mask.npy").astype(bool)
    out = args.output_dir or (root / "figures" / f"phase2_curriculum_comparison_rollout{args.rollout_months}")
    out.mkdir(parents=True, exist_ok=True)

    slots, base = _alignment(forecasts[0].forecast, test_h5)
    leads = np.arange(args.rollout_months + 1)
    n_leads = len(leads)
    n_channels = len(CHANNELS)

    metrics = {}
    for prefix in ["rmse", "acc", "amp_ratio", "var_ratio"]:
        methods = ALL_METHODS if prefix == "rmse" else SKILL_METHODS
        for method in methods:
            metrics[f"{prefix}_{method}"] = np.full((n_leads, n_channels), np.nan, dtype=np.float64)
    spectrum = {f"{method}_{band}": np.full((n_leads, n_channels), np.nan, dtype=np.float64) for method in SKILL_METHODS for band in BANDS}
    nino = {f"rmse_{method}": np.full(n_leads, np.nan, dtype=np.float64) for method in ALL_METHODS}
    nino.update({f"amp_ratio_{method}": np.full(n_leads, np.nan, dtype=np.float64) for method in SKILL_METHODS})
    nino.update({f"cum_corr_{method}": np.full(n_leads, np.nan, dtype=np.float64) for method in SKILL_METHODS})
    nino.update({"truth_mean": np.full(n_leads, np.nan, dtype=np.float64)})
    nino.update({f"{method}_mean": np.full(n_leads, np.nan, dtype=np.float64) for method in MODEL_METHODS})
    nino_traj = {method: np.full((len(slots), n_leads), np.nan, dtype=np.float64) for method in ["truth"] + ALL_METHODS}

    h5s = {spec.name: h5py.File(spec.forecast, "r") for spec in forecasts}
    try:
        with h5py.File(test_h5, "r") as ht:
            channels = _decode_channels(ht)
            if channels != CHANNELS:
                raise ValueError(f"Unexpected channel order: {channels}")
            lat = ht["lat"][:]
            lon = ht["lon"][:]
            nino_mask = _region_mask(lat, lon) & valid_mask[2]
            initial = ht["fields"][base, :, :, :].astype(np.float32)

            for lead in leads:
                target = base + int(lead)
                months = (test_global_start + target) % 12
                truth = ht["fields"][target, :, :, :].astype(np.float32)
                model_fields = _read_forecasts(h5s, slots, int(lead))
                climatology = time_means[months].astype(np.float32)
                fields = {**model_fields, "persistence": initial, "climatology": climatology}
                truth_anom = truth - climatology
                anoms = {method: field - climatology for method, field in fields.items()}

                for ci in range(n_channels):
                    mask = valid_mask[ci]
                    truth_rms = max(_rms(truth_anom[:, ci], mask), 1e-12)
                    truth_var = max(_variance(truth_anom[:, ci], mask), 1e-12)
                    truth_bands = _radial_band_power(truth_anom[:, ci], mask)
                    for method in ALL_METHODS:
                        metrics[f"rmse_{method}"][lead, ci] = _rmse(fields[method][:, ci], truth[:, ci], mask)
                    for method in SKILL_METHODS:
                        metrics[f"acc_{method}"][lead, ci] = _acc(anoms[method][:, ci], truth_anom[:, ci], mask)
                        metrics[f"amp_ratio_{method}"][lead, ci] = _rms(anoms[method][:, ci], mask) / truth_rms
                        metrics[f"var_ratio_{method}"][lead, ci] = _variance(anoms[method][:, ci], mask) / truth_var
                        band_power = _radial_band_power(anoms[method][:, ci], mask)
                        for band in BANDS:
                            spectrum[f"{method}_{band}"][lead, ci] = band_power[band] / max(truth_bands[band], 1e-30)

                clim_nino = np.array([np.nanmean(np.where(nino_mask, time_means[m, 2], np.nan)) for m in months])
                nino_traj["truth"][:, lead] = np.nanmean(np.where(nino_mask, truth[:, 2], np.nan), axis=(1, 2)) - clim_nino
                nino["truth_mean"][lead] = float(np.nanmean(nino_traj["truth"][:, lead]))
                for method in ALL_METHODS:
                    nino_traj[method][:, lead] = np.nanmean(np.where(nino_mask, fields[method][:, 2], np.nan), axis=(1, 2)) - clim_nino
                    nino[f"rmse_{method}"][lead] = float(
                        np.sqrt(np.nanmean((nino_traj[method][:, lead] - nino_traj["truth"][:, lead]) ** 2))
                    )
                truth_nino_rms = max(float(np.sqrt(np.nanmean(nino_traj["truth"][:, lead] ** 2))), 1e-12)
                for method in SKILL_METHODS:
                    nino[f"amp_ratio_{method}"][lead] = float(np.sqrt(np.nanmean(nino_traj[method][:, lead] ** 2)) / truth_nino_rms)
                    nino[f"cum_corr_{method}"][lead] = _corr_flat(
                        nino_traj[method][:, : lead + 1].ravel(),
                        nino_traj["truth"][:, : lead + 1].ravel(),
                    )
                for method in MODEL_METHODS:
                    nino[f"{method}_mean"][lead] = float(np.nanmean(nino_traj[method][:, lead]))
    finally:
        for hf in h5s.values():
            hf.close()

    _plot_rmse_acc(out, leads, metrics)
    _plot_smoothing(out, leads, metrics)
    _plot_spectrum(out, leads, spectrum)
    _plot_nino(out, leads, nino)
    _plot_summary_bars(out, metrics, nino, args.rollout_months)
    _copy_key_pngs(out, args.asset_dir)

    summary = {
        "valid_rollout_slots": slots.astype(int).tolist(),
        "base_test_indices": base.astype(int).tolist(),
        "channels": CHANNELS,
        "methods": ALL_METHODS,
        "model_methods": MODEL_METHODS,
        "bands": BANDS,
        "metrics": {k: v.tolist() for k, v in metrics.items()},
        "spectrum": {k: v.tolist() for k, v in spectrum.items()},
        "nino34": {k: v.tolist() for k, v in nino.items()},
        "nino34_region": {"lat": [-5.0, 5.0], "lon": [190.0, 240.0], "variable": "tos"},
    }
    (out / "phase2_curriculum_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "phase2_curriculum_comparison.npz",
        leads=leads,
        **metrics,
        **{f"spectrum_{k}": v for k, v in spectrum.items()},
        **{f"nino34_{k}": v for k, v in nino.items()},
    )
    print(json.dumps({"output_dir": str(out), "png_count": len(list(out.glob('*.png')))}, indent=2))


if __name__ == "__main__":
    main()
