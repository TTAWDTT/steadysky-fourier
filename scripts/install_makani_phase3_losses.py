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


class RegionalFeatureMMDLoss(FeatureMMDLoss):
    """Batch MMD over regional and optional cross-channel rollout features.

    Phase 6B's global feature MMD was the best current mechanism, but global
    moments can miss basin-scale structure. This variant keeps the same MMD
    contract while adding coarse latitude/longitude regional means and, when
    requested, low-order cross-channel covariance features.
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
        include_regional_mean: bool = True,
        include_cross_channel_covariance: bool = False,
        num_lat_bands: int = 4,
        num_lon_bands: int = 4,
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
            include_mean=include_mean,
            include_log_variance=include_log_variance,
            include_lowpass_mean=include_lowpass_mean,
            bandwidth=bandwidth,
            spatial_distributed=spatial_distributed,
            eps=eps,
        )
        self.include_regional_mean = bool(include_regional_mean)
        self.include_cross_channel_covariance = bool(include_cross_channel_covariance)
        self.num_lat_bands = int(num_lat_bands)
        self.num_lon_bands = int(num_lon_bands)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        feats = [super()._features(x)]
        if self.include_regional_mean:
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                x,
                output_size=(self.num_lat_bands, self.num_lon_bands),
            )
            feats.append(pooled.flatten(start_dim=1))
        if self.include_cross_channel_covariance:
            flat = x.flatten(start_dim=2)
            flat = flat - flat.mean(dim=-1, keepdim=True)
            cov = flat @ flat.transpose(1, 2)
            cov = cov / torch.clamp(torch.tensor(flat.shape[-1], device=x.device, dtype=x.dtype), min=1.0)
            diag = torch.diagonal(cov, dim1=1, dim2=2)
            cov = cov / torch.sqrt(torch.clamp(diag[:, :, None] * diag[:, None, :], min=self.eps))
            feats.append(cov.flatten(start_dim=1))
        return torch.cat(feats, dim=1)


class _SteadySkyLatentAutoencoder(torch.nn.Module):
    """Small weather autoencoder used only for frozen latent losses.

    This is a deliberately lightweight WLA-inspired module. The official WLA
    code uses a transformer autoencoder and binary spherical quantization; for
    this controlled four-variable experiment we keep the same two-stage idea
    but use a compact convolutional encoder that can be trained locally before
    the SFNO run.
    """

    def __init__(self, in_channels: int, latent_channels: int = 32, hidden_channels: int = 64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.GroupNorm(num_groups=1, num_channels=latent_channels),
            torch.nn.GELU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(hidden_channels, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
        x = self.decoder(z)
        if x.shape[-2:] != output_shape:
            x = torch.nn.functional.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.shape[-2:])


class LatentFeatureMMDLoss(FeatureMMDLoss):
    """Batch MMD in a frozen WLA-lite latent space.

    Phase 6B's feature MMD is hand-crafted from means, variance and pooled
    fields. This loss replaces that feature map with a learned encoder trained
    as a small weather autoencoder. It keeps the SFNO architecture unchanged:
    only the training-time distribution statistic changes.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        checkpoint_path: str,
        latent_channels: int = 32,
        hidden_channels: int = 64,
        include_global_stats: bool = True,
        include_latent_mean: bool = True,
        include_latent_std: bool = True,
        include_latent_grid: bool = True,
        reconstruction_weight: float = 0.0,
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
            include_mean=False,
            include_log_variance=False,
            include_lowpass_mean=False,
            bandwidth=bandwidth,
            spatial_distributed=spatial_distributed,
            eps=eps,
        )
        self.checkpoint_path = str(checkpoint_path)
        self.in_channels = len(channel_names)
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.include_global_stats = bool(include_global_stats)
        self.include_latent_mean = bool(include_latent_mean)
        self.include_latent_std = bool(include_latent_std)
        self.include_latent_grid = bool(include_latent_grid)
        self.reconstruction_weight = float(reconstruction_weight)
        self._latent_model = None

    def _load_latent_model(self, device: torch.device) -> _SteadySkyLatentAutoencoder:
        if self._latent_model is None:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            model_config = checkpoint.get("model_config", {})
            self.in_channels = int(model_config.get("in_channels", self.in_channels))
            model = _SteadySkyLatentAutoencoder(
                in_channels=self.in_channels,
                latent_channels=int(model_config.get("latent_channels", self.latent_channels)),
                hidden_channels=int(model_config.get("hidden_channels", self.hidden_channels)),
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._latent_model = model
        return self._latent_model.to(device=device)

    def _lead_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if x.shape[1] == self.in_channels:
            return x, 1
        if x.shape[1] % self.in_channels != 0:
            raise RuntimeError(
                f"LatentFeatureMMDLoss expected {self.in_channels} channels or a multiple, got {x.shape[1]}"
            )
        batch, channels, height, width = x.shape
        leads = channels // self.in_channels
        return x.reshape(batch, leads, self.in_channels, height, width).reshape(batch * leads, self.in_channels, height, width), leads

    def _latent_features(self, x: torch.Tensor) -> torch.Tensor:
        model = self._load_latent_model(x.device)
        x_for_encoder, _ = self._lead_batch(x.to(dtype=torch.float32))
        z = model.encode(x_for_encoder)
        feats = []
        if self.include_global_stats:
            mean = x_for_encoder.mean(dim=(-2, -1))
            centered = x_for_encoder - mean[..., None, None]
            var = centered.square().mean(dim=(-2, -1))
            lowpass = torch.nn.functional.avg_pool2d(x_for_encoder, kernel_size=12, stride=12)
            feats.extend([mean, torch.log(var + self.eps), lowpass.flatten(start_dim=1)])
        if self.include_latent_mean:
            feats.append(z.mean(dim=(-2, -1)))
        if self.include_latent_std:
            z_centered = z - z.mean(dim=(-2, -1), keepdim=True)
            feats.append(torch.log(z_centered.square().mean(dim=(-2, -1)) + self.eps))
        if self.include_latent_grid:
            pooled = torch.nn.functional.adaptive_avg_pool2d(z, output_size=(6, 12))
            feats.append(pooled.flatten(start_dim=1))
        if not feats:
            raise RuntimeError("LatentFeatureMMDLoss needs at least one feature family")
        return torch.cat([f.flatten(start_dim=1) for f in feats], dim=1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self._latent_features(x)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        mmd = super().forward(prd, tar, wgt=wgt, **kwargs)
        if self.reconstruction_weight <= 0:
            return mmd
        model = self._load_latent_model(prd.device)
        pred_float = prd.to(dtype=torch.float32)
        pred_leads, num_leads = self._lead_batch(pred_float)
        recon = model(pred_leads)
        recon_loss = (recon - pred_leads).square().mean(dim=(-2, -1))
        recon_loss = recon_loss.reshape(prd.shape[0], num_leads * self.in_channels)
        if wgt is not None:
            channel_weight = wgt.mean(dim=(-2, -1))
            recon_loss = recon_loss * channel_weight
        return mmd + self.reconstruction_weight * recon_loss.to(dtype=mmd.dtype)


class ShortLeadLpLoss(GeometricBaseLoss):
    """Extra pointwise anchor on the first few rollout leads.

    Makani's built-in tendency transform is single-step oriented and cannot be
    used directly when multistep predictions are flattened as
    (n_future + 1) * channels. This loss instead uses the lead_time_step vector
    supplied by LossHandler to make early rollout field errors more expensive.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        p: float = 2.0,
        squared: bool = True,
        max_lead: int = 2,
        decay: float = 1.0,
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
        self.p = float(p)
        self.squared = bool(squared)
        self.max_lead = int(max_lead)
        self.decay = float(decay)
        self.eps = float(eps)

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        num_examples = prd.shape[0]
        diff = torch.abs(prd - tar).pow(self.p)
        if wgt is not None:
            diff = diff * wgt
        loss = self.quadrature(diff).reshape(num_examples, -1)
        if not self.squared:
            loss = loss.pow(1.0 / self.p)

        lead_time_step = kwargs.get("lead_time_step")
        if lead_time_step is None:
            return loss
        leads = lead_time_step.to(device=loss.device, dtype=loss.dtype)
        mask = (leads <= self.max_lead).to(dtype=loss.dtype)
        weights = torch.exp(-leads / max(self.decay, self.eps)) * mask
        if weights.numel() != loss.shape[1]:
            if loss.shape[1] == self.n_channels:
                weights = torch.ones(loss.shape[1], device=loss.device, dtype=loss.dtype)
            else:
                raise RuntimeError(f"ShortLeadLpLoss got {loss.shape[1]} loss channels but {weights.numel()} lead weights")
        return loss * weights[None, :]


class SpatialMeanDriftLoss(GeometricBaseLoss):
    """Weak invariant penalty for slow spatial-mean drift.

    This loss deliberately ignores pointwise phase. It only compares the
    weighted spatial mean of prediction and target for each example/channel.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
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
        self.eps = float(eps)

    def _weighted_mean(self, x: torch.Tensor, wgt: Optional[torch.Tensor]) -> torch.Tensor:
        if wgt is None:
            return x.mean(dim=(-2, -1))
        weights = wgt
        while weights.ndim < x.ndim:
            weights = weights.unsqueeze(0)
        denom = torch.clamp(weights.sum(dim=(-2, -1)), min=self.eps)
        return (x * weights).sum(dim=(-2, -1)) / denom

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pred_mean = self._weighted_mean(prd, wgt)
        targ_mean = self._weighted_mean(tar, wgt)
        return (pred_mean - targ_mean).square()


class SpectralShapeLoss(GeometricBaseLoss):
    """Match broad normalized spectral shape without prescribing phase.

    Unlike FourierBandLpLoss and SpectralEnergyMatchLoss, this compares only
    the low/mid/high energy proportions. Absolute energy and phase are left to
    the field loss and distribution loss.
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
        high_weight: float = 0.25,
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

    def _shape(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        energy = torch.stack(energies, dim=-1)
        shape = energy / torch.clamp(energy.sum(dim=-1, keepdim=True), min=self.eps)
        return torch.log(shape + self.eps), weights

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pred_shape, weights = self._shape(prd)
        targ_shape, _ = self._shape(tar)
        diff = (pred_shape - targ_shape).square() * weights[None, None, :]
        denom = torch.clamp(weights.sum(), min=self.eps)
        loss = diff.sum(dim=-1) / denom
        if wgt is not None:
            channel_weight = wgt.mean(dim=(-2, -1))
            loss = loss * channel_weight
        return loss
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
    if "RegionalFeatureMMDLoss" not in registry_text:
        extra_imports.append("RegionalFeatureMMDLoss")
    if "LatentFeatureMMDLoss" not in registry_text:
        extra_imports.append("LatentFeatureMMDLoss")
    if "ShortLeadLpLoss" not in registry_text:
        extra_imports.append("ShortLeadLpLoss")
    if "SpatialMeanDriftLoss" not in registry_text:
        extra_imports.append("SpatialMeanDriftLoss")
    if "SpectralShapeLoss" not in registry_text:
        extra_imports.append("SpectralShapeLoss")
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
    if '"regional_feature_mmd": RegionalFeatureMMDLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "regional_feature_mmd": RegionalFeatureMMDLoss,\n')
    if '"latent_feature_mmd": LatentFeatureMMDLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "latent_feature_mmd": LatentFeatureMMDLoss,\n')
    if '"short_lead_l2": ShortLeadLpLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "short_lead_l2": ShortLeadLpLoss,\n')
    if '"spatial_mean_drift": SpatialMeanDriftLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "spatial_mean_drift": SpatialMeanDriftLoss,\n')
    if '"spectral_shape": SpectralShapeLoss' not in registry_text:
        registry_text = registry_text.replace(map_insert, map_insert + '    "spectral_shape": SpectralShapeLoss,\n')
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
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import RegionalFeatureMMDLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import LatentFeatureMMDLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import ShortLeadLpLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import SpatialMeanDriftLoss\n"
    if init_line not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + init_line, encoding="utf-8")
    init_text = init_file.read_text(encoding="utf-8")
    init_line = "from .steadysky_fourier_loss import SpectralShapeLoss\n"
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
