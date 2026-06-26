#!/usr/bin/env python3
"""Count parameters for a Makani config without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from ruamel.yaml import YAML


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--makani-root", required=True, type=Path)
    parser.add_argument("--yaml-config", required=True, type=Path)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.makani_root))

    from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet

    cfg = YAML(typ="safe").load(args.yaml_config.read_text(encoding="utf-8"))
    params = dict(cfg[args.config])
    n_channels = len(params["channel_names"])
    params.update(
        {
            "in_channels": list(range(n_channels)),
            "out_channels": list(range(n_channels)),
            "N_in_channels": n_channels,
            "N_out_channels": n_channels,
            "inp_chans": n_channels,
            "out_chans": n_channels,
            "inp_shape": (params["img_shape_x"], params["img_shape_y"]),
            "out_shape": (params["img_shape_x"], params["img_shape_y"]),
            "enable_nhwc": False,
        }
    )

    model = SphericalFourierNeuralOperatorNet(**params)
    def count_param_entries(include_only_trainable: bool) -> int:
        total = 0
        for p in model.parameters():
            if include_only_trainable and not p.requires_grad:
                continue
            view = torch.view_as_real(p) if p.is_complex() else p
            total += view.numel()
        return total

    trainable = count_param_entries(include_only_trainable=True)
    total = count_param_entries(include_only_trainable=False)
    payload = {
        "config": args.config,
        "nettype": params.get("nettype"),
        "img_shape_x": params.get("img_shape_x"),
        "img_shape_y": params.get("img_shape_y"),
        "channels": n_channels,
        "embed_dim": params.get("embed_dim"),
        "num_layers": params.get("num_layers"),
        "scale_factor": params.get("scale_factor"),
        "operator_type": params.get("operator_type"),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
