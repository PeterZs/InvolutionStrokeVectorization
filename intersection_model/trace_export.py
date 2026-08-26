#!/usr/bin/env python
"""Standalone script to trace an intersection model and export it as TorchScript.
"""
import argparse

import torch

import litmodel
import train
from runconfig import RunConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace an intersection model checkpoint and export it as TorchScript."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .ckpt file")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model class name (must exist in mymodel)",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default="output_traced.pt",
        help="Output path for the traced TorchScript file",
    )
    parser.add_argument(
        "--trace_device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Device used to trace the model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = RunConfig(
        mode="trace",
        model=args.model,
        load_from=args.checkpoint,
        save_to=args.save_to,
        trace_device=args.trace_device,
    )

    model, _ckpt_path, _previous_run_id = litmodel.load_lightning_model(config)

    train.trace_for_export(config.save_to, model, torch.device(config.trace_device))
    print(f"saved traced model to {config.save_to}")


if __name__ == "__main__":
    main()
