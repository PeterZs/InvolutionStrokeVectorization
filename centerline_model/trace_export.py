#!/usr/bin/env python
"""Standalone script to trace a centerline model and export it to TorchScript.

Decoupled from main.py: no env variables, no wandb, no dataset loading.
"""
import argparse
from types import SimpleNamespace

import utils
import train
import mymodel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace a centerline model checkpoint and export it as TorchScript."
    )
    parser.add_argument("model", type=str, help="Model class name (must exist in mymodel)")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("save_to", type=str, help="Output path for the traced TorchScript file")
    parser.add_argument(
        "--image_size",
        type=int,
        default=592,
        help="Spatial size of the example input used for tracing (default: 592)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Minimal namespace compatible with utils.load_model_checkpoint.
    ckpt_args = SimpleNamespace(
        model=args.model,
        checkpoint=args.checkpoint,
        latest=False,
    )
    model, previous_state_dict = utils.load_model_checkpoint(ckpt_args, mymodel)

    # trace_for_export only reads config.image_size on this path.
    config = SimpleNamespace(image_size=args.image_size)

    train.trace_for_export(args.save_to, model, config, previous_state_dict)
    print(f"saved traced model to {args.save_to}")


if __name__ == "__main__":
    main()
