#!/usr/bin/env python3
"""Check Makani training logs for early-stop safety conditions."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
METRIC_PATTERNS = {
    "training_loss": re.compile(rf"training loss:\s*({NUMBER})", re.IGNORECASE),
    "validation_loss": re.compile(rf"validation loss:\s*({NUMBER})", re.IGNORECASE),
    "gradient_norm": re.compile(rf"gradient norm:\s*({NUMBER})", re.IGNORECASE),
}


def parse_number(text: str) -> float:
    return float(text.lower())


def iter_metrics(paths: list[Path]) -> list[tuple[str, float, Path, int]]:
    metrics: list[tuple[str, float, Path, int]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                for name, pattern in METRIC_PATTERNS.items():
                    match = pattern.search(line)
                    if match:
                        metrics.append((name, parse_number(match.group(1)), path, line_no))
    return metrics


def patience_triggered(values: list[float], min_points: int, patience: int, min_delta: float) -> bool:
    if patience <= 0 or len(values) < max(min_points, patience + 1):
        return False

    best = math.inf
    stale = 0
    for index, value in enumerate(values, start=1):
        if index < min_points:
            if value < best:
                best = value
            continue
        if value < best - min_delta:
            best = value
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Makani log files to scan in chronological order.")
    parser.add_argument("--patience", type=int, default=8, help="Validation points without improvement before stopping.")
    parser.add_argument("--min-points", type=int, default=20, help="Minimum validation points before plateau early stop can trigger.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Required validation-loss improvement.")
    parser.add_argument("--max-valid-loss", type=float, default=1e6, help="Validation loss above this is treated as divergence.")
    args = parser.parse_args()

    metrics = iter_metrics(args.logs)
    if not metrics:
        print("EARLY_STOP_CHECK no parseable Makani metrics found; continuing.", file=sys.stderr)
        return 0

    for name, value, path, line_no in metrics:
        if not math.isfinite(value):
            print(f"EARLY_STOP non-finite {name}={value} at {path}:{line_no}", file=sys.stderr)
            return 10

    valid_losses = [value for name, value, _, _ in metrics if name == "validation_loss"]
    if valid_losses:
        latest = valid_losses[-1]
        best = min(valid_losses)
        print(
            "EARLY_STOP_CHECK "
            f"validation_points={len(valid_losses)} best={best:.8g} latest={latest:.8g}",
            file=sys.stderr,
        )
        if latest > args.max_valid_loss:
            print(
                f"EARLY_STOP validation loss {latest:.8g} exceeds threshold {args.max_valid_loss:.8g}",
                file=sys.stderr,
            )
            return 11
        if patience_triggered(valid_losses, args.min_points, args.patience, args.min_delta):
            print(
                "EARLY_STOP validation loss plateau "
                f"(patience={args.patience}, min_points={args.min_points}, min_delta={args.min_delta})",
                file=sys.stderr,
            )
            return 12
    else:
        print("EARLY_STOP_CHECK no validation loss found; continuing.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
