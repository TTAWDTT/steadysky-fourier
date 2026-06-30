#!/usr/bin/env python3
"""Install SteadySky Phase-3 losses into a local Makani checkout.

The project intentionally keeps Makani as an external dependency. This helper
adds a small custom loss module and registers it in Makani's loss registry so
Phase-3 experiments can be reproduced from this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path


LOSS_MODULE = '''# SPDX-License-Identifier: Apache-2.0
"""SteadySky Fourier curriculum losses for Makani."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from makani.utils.losses.base_loss import GeometricBaseLoss


class FourierBandLpLoss(GeometricBaseLoss):
    """Squared 2D Fourier loss with configurable radial band weights.

    The loss returns one value per example and channel, matching Makani's
    deterministic loss contract. It is intentionally local to each latitude-
    longitude plane and does not alter the model architecture.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        low_weight: float = 1.0,
        mid_weight: float = 0.0,
        high_weight: float = 0.0,
        low_max: float = 5.0,
        mid_max: float = 20.0,
        include_zero: bool = True,
        spatial_distributed: Optional[bool] = False,
        eps: float = 1.0e-12,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )
        self.low_weight = float(low_weight)
        self.mid_weight = float(mid_weight)
        self.high_weight = float(high_weight)
        self.low_max = float(low_max)
        self.mid_max = float(mid_max)
        self.include_zero = bool(include_zero)
        self.eps = float(eps)
        self.register_buffer("band_weight", torch.empty(0), persistent=False)

    def _band_weight(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.band_weight.numel() and self.band_weight.shape == (height, width // 2 + 1):
            return self.band_weight.to(device=device, dtype=dtype)

        ky = torch.fft.fftfreq(height, device=device)[:, None] * height
        kx = torch.fft.rfftfreq(width, device=device)[None, :] * width
        kr = torch.sqrt(kx * kx + ky * ky)
        weight = torch.zeros((height, width // 2 + 1), device=device, dtype=dtype)

        low_mask = kr <= self.low_max
        if not self.include_zero:
            low_mask = low_mask & (kr > 0)
        mid_mask = (kr > self.low_max) & (kr <= self.mid_max)
        high_mask = kr > self.mid_max
        weight = torch.where(low_mask, torch.as_tensor(self.low_weight, device=device, dtype=dtype), weight)
        weight = torch.where(mid_mask, torch.as_tensor(self.mid_weight, device=device, dtype=dtype), weight)
        weight = torch.where(high_mask, torch.as_tensor(self.high_weight, device=device, dtype=dtype), weight)

        self.band_weight = weight.detach()
        return weight

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        diff = prd - tar
        if wgt is not None:
            diff = diff * wgt

        fft_dtype = torch.float32 if diff.dtype in (torch.float16, torch.bfloat16) else diff.dtype
        diff = diff.to(dtype=fft_dtype)
        height, width = diff.shape[-2:]
        weight = self._band_weight(height, width, diff.device, fft_dtype)
        coeff = torch.fft.rfft2(diff, dim=(-2, -1), norm="ortho")
        power = coeff.real.square() + coeff.imag.square()
        weighted_power = power * weight[None, None, :, :]
        denom = torch.clamp(weight.sum(), min=self.eps)
        return weighted_power.sum(dim=(-2, -1)) / denom


class SpectralEnergyMatchLoss(GeometricBaseLoss):
    """Match target anomaly energy without prescribing Fourier phase.

    This loss compares log spectral energy between prediction and target in
    broad radial bands. It is designed as a small rollout-stage regularizer:
    field L2 keeps the forecast in phase where possible, while this term only
    discourages collapse toward a low-energy climatological attractor.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        low_weight: float = 1.0,
        mid_weight: float = 1.0,
        high_weight: float = 0.0,
        low_max: float = 5.0,
        mid_max: float = 20.0,
        remove_spatial_mean: bool = True,
        spatial_distributed: Optional[bool] = False,
        eps: float = 1.0e-8,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )
        self.low_weight = float(low_weight)
        self.mid_weight = float(mid_weight)
        self.high_weight = float(high_weight)
        self.low_max = float(low_max)
        self.mid_max = float(mid_max)
        self.remove_spatial_mean = bool(remove_spatial_mean)
        self.eps = float(eps)
        self.register_buffer("band_masks", torch.empty(0), persistent=False)
        self.register_buffer("band_weights", torch.empty(0), persistent=False)

    def _bands(self, height: int, width: int, device: torch.device, dtype: torch.dtype):
        expected = (3, height, width // 2 + 1)
        if self.band_masks.numel() and tuple(self.band_masks.shape) == expected:
            return self.band_masks.to(device=device, dtype=torch.bool), self.band_weights.to(device=device, dtype=dtype)

        ky = torch.fft.fftfreq(height, device=device)[:, None] * height
        kx = torch.fft.rfftfreq(width, device=device)[None, :] * width
        kr = torch.sqrt(kx * kx + ky * ky)
        masks = torch.stack(
            [
                kr <= self.low_max,
                (kr > self.low_max) & (kr <= self.mid_max),
                kr > self.mid_max,
            ],
            dim=0,
        )
        weights = torch.as_tensor([self.low_weight, self.mid_weight, self.high_weight], device=device, dtype=dtype)
        self.band_masks = masks.detach()
        self.band_weights = weights.detach()
        return masks, weights

    def _log_band_energy(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove_spatial_mean:
            x = x - x.mean(dim=(-2, -1), keepdim=True)
        fft_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x = x.to(dtype=fft_dtype)
        height, width = x.shape[-2:]
        masks, weights = self._bands(height, width, x.device, fft_dtype)
        coeff = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        power = coeff.real.square() + coeff.imag.square()
        energies = []
        for mask in masks:
            denom = torch.clamp(mask.sum().to(dtype=fft_dtype), min=1.0)
            energies.append((power * mask[None, None]).sum(dim=(-2, -1)) / denom)
        return torch.log(torch.stack(energies, dim=-1) + self.eps), weights

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pred_log, weights = self._log_band_energy(prd)
        targ_log, _ = self._log_band_energy(tar)
        diff = (pred_log - targ_log).square() * weights[None, None, :]
        denom = torch.clamp(weights.sum(), min=self.eps)
        loss = diff.sum(dim=-1) / denom
        if wgt is not None:
            channel_weight = wgt.mean(dim=(-2, -1))
            loss = loss * channel_weight
        return loss


class AttractorStatsLoss(GeometricBaseLoss):
    """Match coarse attractor statistics without matching spatial phase.

    This loss is intentionally weaker than pointwise rollout loss. It compares
    per-field spatial mean, variance, and adjacent-channel covariance. The goal
    is to make the collapsed low-energy attractor expensive while avoiding the
    Phase-3 failure mode of prescribing exact Fourier phase.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        mean_weight: float = 1.0,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.25,
        use_log_variance: bool = True,
        spatial_distributed: Optional[bool] = False,
        eps: float = 1.0e-8,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )
        self.mean_weight = float(mean_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.use_log_variance = bool(use_log_variance)
        self.eps = float(eps)

    def _stats(self, x: torch.Tensor):
        mean = x.mean(dim=(-2, -1))
        centered = x - mean[..., None, None]
        var = centered.square().mean(dim=(-2, -1))
        if x.shape[1] > 1:
            cov = (centered[:, :-1] * centered[:, 1:]).mean(dim=(-2, -1))
        else:
            cov = None
        return mean, var, cov

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pred_mean, pred_var, pred_cov = self._stats(prd)
        targ_mean, targ_var, targ_cov = self._stats(tar)

        mean_loss = (pred_mean - targ_mean).square()
        if self.use_log_variance:
            var_loss = (torch.log(pred_var + self.eps) - torch.log(targ_var + self.eps)).square()
        else:
            var_loss = (pred_var - targ_var).square() / (targ_var.detach() + self.eps)
        loss = self.mean_weight * mean_loss + self.variance_weight * var_loss

        if pred_cov is not None and targ_cov is not None and self.covariance_weight > 0:
            cov_loss = (pred_cov - targ_cov).square() / (targ_var[:, :-1].detach() * targ_var[:, 1:].detach() + self.eps).sqrt()
            cov_pad = torch.zeros_like(loss)
            cov_pad[:, :-1] = cov_pad[:, :-1] + 0.5 * cov_loss
            cov_pad[:, 1:] = cov_pad[:, 1:] + 0.5 * cov_loss
            loss = loss + self.covariance_weight * cov_pad

        if wgt is not None:
            channel_weight = wgt.mean(dim=(-2, -1))
            loss = loss * channel_weight
        return loss


class FeatureMMDLoss(GeometricBaseLoss):
    """Batch-level distribution matching over coarse field features.

    Unlike AttractorStatsLoss, this loss does not compare sample i to sample i.
    It compares the batch distribution of prediction features with the batch
    distribution of target features using an RBF-kernel MMD. This is a minimal
    invariant-measure proxy that keeps the architecture fixed.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        include_mean: bool = True,
        include_log_variance: bool = True,
        include_lowpass_mean: bool = True,
        bandwidth: float = 1.0,
        spatial_distributed: Optional[bool] = False,
        eps: float = 1.0e-8,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )
        self.include_mean = bool(include_mean)
        self.include_log_variance = bool(include_log_variance)
        self.include_lowpass_mean = bool(include_lowpass_mean)
        self.bandwidth = float(bandwidth)
        self.eps = float(eps)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        mean = x.mean(dim=(-2, -1))
        centered = x - mean[..., None, None]
        if self.include_mean:
            feats.append(mean)
        if self.include_log_variance:
            var = centered.square().mean(dim=(-2, -1))
            feats.append(torch.log(var + self.eps))
        if self.include_lowpass_mean:
            pooled = torch.nn.functional.avg_pool2d(x, kernel_size=12, stride=12)
            feats.append(pooled.flatten(start_dim=1))
        if not feats:
            raise RuntimeError("FeatureMMDLoss needs at least one feature family")
        return torch.cat([f.flatten(start_dim=1) for f in feats], dim=1)

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = x.square().sum(dim=1, keepdim=True)
        y2 = y.square().sum(dim=1, keepdim=True).transpose(0, 1)
        dist2 = torch.clamp(x2 + y2 - 2.0 * x @ y.transpose(0, 1), min=0.0)
        gamma = 1.0 / max(2.0 * self.bandwidth * self.bandwidth, self.eps)
        return torch.exp(-gamma * dist2)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pred_features = self._features(prd)
        targ_features = self._features(tar)
        pred_features = (pred_features - pred_features.mean(dim=0, keepdim=True)) / (pred_features.std(dim=0, keepdim=True) + self.eps)
        targ_features = (targ_features - targ_features.mean(dim=0, keepdim=True)) / (targ_features.std(dim=0, keepdim=True) + self.eps)
        k_pp = self._kernel(pred_features, pred_features).mean()
        k_tt = self._kernel(targ_features, targ_features).mean()
        k_pt = self._kernel(pred_features, targ_features).mean()
        mmd = torch.clamp(k_pp + k_tt - 2.0 * k_pt, min=0.0)
        return mmd.expand(prd.shape[0], prd.shape[1])
'''


def install(makani_root: Path) -> None:
    loss_dir = makani_root / "makani" / "utils" / "losses"
    registry = makani_root / "makani" / "utils" / "loss.py"
    init_file = loss_dir / "__init__.py"
    if not loss_dir.is_dir() or not registry.exists() or not init_file.exists():
        raise FileNotFoundError(f"{makani_root} does not look like a Makani checkout")

    (loss_dir / "steadysky_fourier_loss.py").write_text(LOSS_MODULE, encoding="utf-8")

    registry_text = registry.read_text(encoding="utf-8")
    marker = "from .losses import DriftRegularization, HydrostaticBalanceLoss, SpectralRegularization\n"
    extra_imports = []
    if "FourierBandLpLoss" not in registry_text:
        extra_imports.append("FourierBandLpLoss")
    if "SpectralEnergyMatchLoss" not in registry_text:
        extra_imports.append("SpectralEnergyMatchLoss")
    if "AttractorStatsLoss" not in registry_text:
        extra_imports.append("AttractorStatsLoss")
    if "FeatureMMDLoss" not in registry_text:
        extra_imports.append("FeatureMMDLoss")
    if extra_imports:
        registry_text = registry_text.replace(marker, marker + f"from .losses import {', '.join(extra_imports)}\n")

    map_insert = '    "spectral_regularization": SpectralRegularization,\n'
    if '"fourier2d": FourierBandLpLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "fourier2d": FourierBandLpLoss,\n')
    if '"spectral_energy_match": SpectralEnergyMatchLoss' not in registry_text:
        registry_text = registry_text.replace(
            map_insert,
            map_insert + '    "spectral_energy_match": SpectralEnergyMatchLoss,\n',
        )
    if '"attractor_stats": AttractorStatsLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "attractor_stats": AttractorStatsLoss,\n')
    if '"feature_mmd": FeatureMMDLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "feature_mmd": FeatureMMDLoss,\n')
    registry.write_text(registry_text, encoding="utf-8")

    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import FourierBandLpLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import SpectralEnergyMatchLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import AttractorStatsLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import FeatureMMDLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--makani-root", type=Path, required=True)
    args = parser.parse_args()
    install(args.makani_root)
    print(f"installed Phase-3 losses into {args.makani_root}")


if __name__ == "__main__":
    main()
