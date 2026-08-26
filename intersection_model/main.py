#!/usr/bin/env python
import os
import argparse
import torch
import wandb

import utils
import train
import dataset
import mymodel
import pprint
from pathlib import Path
from datetime import datetime
from runconfig import RunConfig
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

import litmodel


def parse_args_and_get_config():
    parser = argparse.ArgumentParser(description="Run training or evaluation")

    parser.add_argument(
        "--mode",
        choices=["train", "test", "predict", "trace"],
        required=True,
        help="Execution mode: 'train' or 'test'."
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="List of dataset names to use. If empty, all available datasets are used."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Select model to use"
    )
    
    parser.add_argument(
        "--debug_run",
        type=int,
        default=0,
        help='''
        If specified and >=0, train or eval with only `debug_run` samples for debugging purposes.
        Note that in eval mode, this will use `debug_run` samples for *each* specified dataset.
        '''
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="If specified, train with cpu. Otherwise, main will throw an error if cuda is not available."
    )
    
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="List of images to predict on."
    )
    
    parser.add_argument(
        "--pred_output_dir",
        type=str,
        default=".",
        help="Output directory for predictions."
    )
    
    parser.add_argument(
        "--save_to",
        type=str,
        default=".",
        help="Output directory for exporting the jit traced model"
    )
    
    parser.add_argument(
         "--trace_device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="device used to trace the model"
    )

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_output_every", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--latest", action="store_true", default=False)
    parser.add_argument("--latest-best", action="store_true", default=False)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--no_shuffle", action="store_true", default=False)
    parser.add_argument("--use_bf16", action="store_true", default=False)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--patience_early_stop", type=int, default=20)
    parser.add_argument("--no_checkpointing", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument(
        "--loss_funcs",
        type=str,
        default="MaskedCrossEntropyLoss:1.0",
        help="Comma-separated list of loss functions and weights, e.g. 'MSELoss:1.0,L1Loss:0.5'"
    )

    args = parser.parse_args()
    args_dict = vars(args)
    
    # Convert loss_funcs string to dictionary
    loss_dict = utils.parse_loss_funcs(args.loss_funcs)
    args_dict["loss_funcs"] = loss_dict
    
    if args_dict["run_id"] is None:
        args_dict["run_id"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    config = RunConfig(**args_dict)
    
    assert config.mode in ["predict", "trace"] or len(config.datasets), "must provide at least 1 dataset"
    
    assert config.mode != "predict" or (
        config.mode=="predict" and len(config.images)), "in predict mode, you must specify at least 1 image to predict"
    
    if config.debug_run<=0 and config.no_shuffle:
        raise ValueError("shuffling is mandatory for training")
    
    if config.debug_run <=0 and config.val_split > 0.1:
        raise ValueError("val_split value is too large")
    
    if not torch.cuda.is_available() and not config.cpu and not config.mode == 'trace': 
        raise RuntimeError("default is GPU training/inference but no GPU device found")
    if config.resume and (
        config.load_from is None and not config.latest and not config.latest_best):
        raise ValueError("specified to resume, but no checkpoint provided")
    if config.mode in ['test', 'trace'] and (
        config.load_from is None and not config.latest and not config.latest_best):
        raise ValueError("specified test or trace mode, but you didn't provide a checkpoint")
    
    assert not config.no_checkpointing or config.no_checkpointing and config.debug_run >0

    assert config.image_size % 8 == 0
    
    return config

if __name__ == "__main__":
    
    config: RunConfig = parse_args_and_get_config()    
    # Load environment variables
    utils.load_env()
    
    # Set seeds for reproducibility
    utils.set_seed(config.seed)
    
    utils.prepare_system()
    
    #Load model
    model, ckpt_path, previous_run_id =litmodel.load_lightning_model(config)
    print("previous id:", previous_run_id)
    # Initialize wandb
    if not os.environ.get("WANDB_PROJECT", None) or config.mode == 'trace':
        print("wandb logging disabled")
    elif utils.get_rank() == 0:
        name, tags = utils.get_run_name_tags(config, previous_run_id)
        wandb_pr = os.environ.get("WANDB_PROJECT")
        wandb.init(project=wandb_pr, id = None,tags=tags,name=name)
        print(f"WandB logging initialized for project: {wandb_pr}")
        #add complete run parameters
        wandb.config.update(vars(config))

    if utils.get_rank() == 0:
        #print complete run parameters to stdout
        pprint.pprint(vars(config))
    
    # Get dataset directories
    scratch_path = os.environ.get("DATA_ROOT")
    datasets= {x: f"{scratch_path}/{config.mode}/{x}" for x in config.datasets}
    print(datasets)
    
    if config.mode == 'train':
        train_data, val_data = dataset.get_dataloaders(datasets, 
                                                       batch_size=config.batch_size, 
                                                       tiny_set=config.debug_run, 
                                                       val_split=config.val_split,
                                                       seed=config.seed,
                                                       num_workers=0)
        train.train_model(train_data, val_data, model, config, ckpt_path)

    if config.mode == 'test':
        test_datasets = [dataset.get_dataloaders({k: v}, 
                                            batch_size=config.batch_size, 
                                            tiny_set=config.debug_run, 
                                            val_split=0,
                                            seed=config.seed,
                                            shuffle=False,
                                            num_workers=0)[0]
                         for k, v in datasets.items()]
        train.test_model(test_datasets, model, config)
        
    if config.mode == 'predict':
        model.eval()
        raise ValueError("Not implemented")
    
    if config.mode == 'trace':
        train.trace_for_export(config.save_to, model, torch.device(config.trace_device))

    # Finish wandb run
    if os.environ.get("WANDB_PROJECT", None):
        wandb.finish()