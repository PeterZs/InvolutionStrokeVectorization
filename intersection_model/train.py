import torch
from torch.utils.data import DataLoader
from collections import Counter
import metrics_and_losses
import metrics_and_losses
from contextlib import nullcontext
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback
from runconfig import RunConfig
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection
from torch.export import export, ExportedProgram

import os
import pytorch_lightning as pl
import torch
import wandb
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import json

# Assume utils.get_nn_module_subclasses is available as in your snippet
import utils 
import litmodel
import permutations
import mymodel




def train_model(train_loader_batched, 
                val_loader, 
                lit_model: litmodel.LitModelWrapper, 
                config: RunConfig,
                ckpt_path: str | None
                ):
    
    print("starting training")
    
    precision = "bf16-mixed" if config.use_bf16 else 32
    
    callbacks = []
    if not config.no_checkpointing:
        callbacks = litmodel.get_checkpoint_callbacks(config.model, config.run_id)
    
    
    if val_loader and config.patience_early_stop >0:
        # 1. Early Stopping Callback
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=config.patience_early_stop,
            verbose=True,
            mode="min"
        ))
    
    if config.debug_run > 0:
        callbacks.append(utils.CheckParamsOnTrain())

    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',  
        num_nodes=1,
        strategy='auto',
        use_distributed_sampler=True,
        max_epochs=config.max_epochs,
        precision=precision,
        enable_checkpointing=True, # true because we use ModelCheckpoint callback
        logger=False,
        # validate every epoch is now the default/forced behavior
        check_val_every_n_epoch=1 if val_loader else None,
        callbacks=callbacks,
        enable_progress_bar=False,
        gradient_clip_val=config.grad_clip
    )
    print(utils.trainer_strat(trainer))


    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader_batched,
        val_dataloaders=val_loader,
        #used when resuming, which should set the optimizer states.
        # When loading from a checkpoint without resuming (eg finetuning)
        # this is None
        ckpt_path= ckpt_path if config.resume else None 
    )
    
def test_model(test_loaders: list, 
                lit_model: litmodel.LitModelWrapper, 
                config: RunConfig):
    
    
    precision = "bf16-mixed" if config.use_bf16 else 32
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',  
        num_nodes=1,
        strategy='auto',
        precision=precision,
        logger=False,
        # validate every epoch is now the default/forced behavior
        enable_progress_bar=False,
    )
    
    print("test")
    trainer.test(model=lit_model, dataloaders=test_loaders)
    

def trace_for_export(out_path, lit_model: litmodel.LitModelWrapper, device: torch.device):
    lit_model = lit_model.to(device)
    lit_model.eval()
    randimg = torch.randn(3, 500, 500)
    num_features = 20
    indexes = torch.arange(1, num_features+1)
    
    randseg = permutations.scatter_with_repeats(indexes, 500*500).reshape((500, 500)).long()
    assert len(torch.unique(randseg)) == num_features+1
    randR = (mymodel.symmetric_rand((num_features+1, num_features+1)) > 0.5).to(torch.float32)
    randR[0, 0] = 1
    K = lit_model.config.model_config["max_objects"]
    randR_padded = torch.zeros((K, K), dtype=torch.float32)
    randR_padded[:num_features+1, :num_features+1] = randR
    print("img dtype", randimg.dtype)
    
    lines_rand = torch.randn(lit_model.model.max_objects, 100, 2)
    batch = {"img" : randimg.unsqueeze(0),
             "seg": randseg.unsqueeze(0),
             "lines":lines_rand.unsqueeze(0),
             "R" : randR_padded.unsqueeze(0),
             "n_objects": torch.tensor([num_features]),
    }
    batch = {k: v.to(device) for k, v in batch.items()}
    print("torchscript tracing...")
    # prog: ExportedProgram = export(lit_model, args=(batch, ))
    # torch.export.save(prog, 'output_cpu4.pt2')
    
    script = lit_model.to_torchscript(
        method="trace", 
        example_inputs=batch
        )
    # r = lit_model.model(batch["img"], batch["seg"], batch["R"], batch["n_objects"])
    # script = torch.jit.script(lit_model.model)
    torch.jit.save(script, out_path)