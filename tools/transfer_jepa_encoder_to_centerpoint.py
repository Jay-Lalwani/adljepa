#!/usr/bin/env python3
"""Extract AD-L-JEPA context encoder weights for CenterPoint fine-tuning."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path, help="AD-L-JEPA checkpoint")
    parser.add_argument("--dst", required=True, type=Path, help="Output checkpoint containing detector-compatible keys")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.src, map_location="cpu")
    src_state = ckpt.get("model_state", ckpt)
    dst_state = {}

    if "global_step" in src_state:
        dst_state["global_step"] = src_state["global_step"]

    copied = []
    skipped = []
    for key, value in src_state.items():
        if not key.startswith("backbone_3d.encoder."):
            skipped.append(key)
            continue
        new_key = key.replace("backbone_3d.encoder.", "backbone_3d.", 1)
        dst_state[new_key] = value
        copied.append((key, new_key, tuple(value.shape) if hasattr(value, "shape") else ()))

    if not copied:
        raise RuntimeError(f"No backbone_3d.encoder.* keys found in {args.src}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": dst_state,
            "source_checkpoint": str(args.src),
            "transfer_type": "ad_l_jepa_encoder_to_centerpoint",
        },
        args.dst,
    )
    print(f"wrote {args.dst}")
    print(f"copied encoder tensors: {len(copied)}")
    print(f"dropped pretraining-only tensors: {len(skipped)}")
    for old, new, shape in copied[:12]:
        print(f"  {old} -> {new} {shape}")
    if len(copied) > 12:
        print(f"  ... {len(copied) - 12} more")


if __name__ == "__main__":
    main()
