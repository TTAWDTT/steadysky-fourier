#!/usr/bin/env python3
"""Train a lightweight WLA-inspired autoencoder for Phase 10 latent losses.

The official WLA repository currently publishes model code but not the full
training pipeline or pretrained weights. This script keeps the relevant idea
for SteadySky: learn a compact weather-state latent representation first, then
freeze its encoder during SFNO rollout training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class WalkerH5Dataset(Dataset):
    def __init__(self, h5_path: Path, mean_path: Path, std_path: Path, max_samples: int | None = None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as hf:
            self.length = int(hf["fields"].shape[0])
        self.indices = np.arange(self.length, dtype=np.int64)
        if max_samples is not None and max_samples < self.length:
            self.indices = self.indices[:max_samples]
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.maximum(np.load(std_path).astype(np.float32), 1.0e-6)
        self._h5 = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _fields(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5["fields"]

    def __getitem__(self, item: int) -> torch.Tensor:
        idx = int(self.indices[item])
        x = self._fields()[idx].astype(np.float32)
        x = (x - self.mean.reshape(4, 1, 1)) / self.std.reshape(4, 1, 1)
        return torch.from_numpy(x)


class WLALiteAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 4, latent_channels: int = 32, hidden_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=latent_channels),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
        x = self.decoder(z)
        if x.shape[-2:] != output_shape:
            x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.shape[-2:])


def spectral_shape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)
    pred_power = torch.fft.rfft2(pred.float(), dim=(-2, -1), norm="ortho").abs().square()
    targ_power = torch.fft.rfft2(target.float(), dim=(-2, -1), norm="ortho").abs().square()
    pred_shape = pred_power / torch.clamp(pred_power.sum(dim=(-2, -1), keepdim=True), min=eps)
    targ_shape = targ_power / torch.clamp(targ_power.sum(dim=(-2, -1), keepdim=True), min=eps)
    return (torch.log(pred_shape + eps) - torch.log(targ_shape + eps)).square().mean()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    train_ds = WalkerH5Dataset(
        data_root / "train_raw" / "train.h5",
        data_root / "stats_raw" / "global_means.npy",
        data_root / "stats_raw" / "global_stds.npy",
        max_samples=args.max_train_samples,
    )
    valid_ds = WalkerH5Dataset(
        data_root / "valid_raw" / "valid.h5",
        data_root / "stats_raw" / "global_means.npy",
        data_root / "stats_raw" / "global_stds.npy",
        max_samples=args.max_valid_samples,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = WLALiteAutoencoder(latent_channels=args.latent_channels, hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "wla_lite_best.pt"
    last_path = out_dir / "wla_lite_last.pt"
    history = []
    best_valid = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda", dtype=torch.bfloat16):
                recon = model(x)
                loss = F.mse_loss(recon, x)
                if args.spectral_weight > 0:
                    loss = loss + args.spectral_weight * spectral_shape_loss(recon, x)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())
        train_loss /= max(len(train_loader), 1)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x in valid_loader:
                x = x.to(device=device, non_blocking=True)
                recon = model(x)
                loss = F.mse_loss(recon, x)
                valid_loss += float(loss.detach().cpu())
        valid_loss /= max(len(valid_loader), 1)
        row = {"epoch": epoch, "train_loss": train_loss, "valid_mse": valid_loss}
        history.append(row)
        print(json.dumps(row), flush=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": 4,
                "latent_channels": args.latent_channels,
                "hidden_channels": args.hidden_channels,
            },
            "normalization": "zscore_from_walker_ocean_1deg_full_stats_raw",
            "epoch": epoch,
            "valid_mse": valid_loss,
            "history": history,
        }
        torch.save(checkpoint, last_path)
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(checkpoint, best_path)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved best checkpoint: {best_path}")


def main() -> None:
    default_root = os.environ.get("STEADYSKY_WORK")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=f"{default_root}/data/walker_ocean_1deg_full" if default_root else None)
    parser.add_argument("--output-dir", type=str, default=f"{default_root}/latent/wla_lite" if default_root else "latent/wla_lite")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--spectral-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root is required when STEADYSKY_WORK is not set")
    train(args)


if __name__ == "__main__":
    main()
