import os
import pytorch_lightning as pl
import torch
import wandb
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import json
from runconfig import RunConfig
import utils
import mymodel
from collections import Counter
import metrics_and_losses
from torchmetrics import MetricCollection

from typing import Any
import torch.nn as nn
import cv2
import numpy as np

torch.autograd.set_detect_anomaly(True)


class LitModelWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module, config: RunConfig):
        super().__init__()
        # 1. Save all init args to self.hparams automatically. 
        # This makes it save the config.
        # we do NOT include the model because we only want its weights
        # which is handled by state_dict()
        #self.save_hyperparameters(ignore=["model"])

        self.model = model
        
        self.config = config
        
        #set in setup() because it needs device arg
        self.loss_fn_wrapper = None
        
        self.epoch_loss_components = Counter()
        self.val_loss_func =metrics_and_losses.LOSS_MAP["MaskedCrossEntropyLoss"]
        
        #set in on_load_checkpoint
        self.previous_run_id = None
        
        #metrics to use for every dataset
        self.metric_template = MetricCollection({
            'ce_metric': metrics_and_losses.GlobalStatsMetric(
               metrics_and_losses.ce_metric, smaller_is_better=True),
            'correct_connections': metrics_and_losses.GlobalStatsMetric(
                metrics_and_losses.correct_connections, smaller_is_better=False),
        })
        
        self.epoch_example_data = None
        self.val_example_data = None
    
    def setup(self, stage: str):
        self.loss_fn_wrapper = metrics_and_losses.build_loss_func(self.config.loss_funcs, self.device)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    
    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        #return self.model(batch["img"], batch["seg"], batch["R"], batch["n_objects"])
        return self.model(batch)
    # def forward(self, img: torch.Tensor, seg_mask: torch.Tensor, constraint_matrix: torch.Tensor, n_objects_int: torch.Tensor) -> Any:
        # return self.model(img, seg_mask, constraint_matrix, n_objects_int)
    
    def on_train_start(self):
        if self.epoch_example_data is None:
            print("get examples for output show")
            train_loader = self.trainer.train_dataloader
            self.epoch_example_data = next(iter(train_loader))
        if self.val_example_data is None and self.trainer.val_dataloaders is not None:
            val_loader = self.trainer.val_dataloaders
            self.val_example_data = next(iter(val_loader))
    
    def _get_example_data(self, val = False):
        batch = None
        if val and self.val_example_data is not None:
            batch = self.val_example_data
        if not val and self.epoch_example_data is not None:
            batch= self.epoch_example_data
        
        if batch is not None:
            return {k: v.to(
            self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return None

    def on_fit_start(self):
        print("stored run id", self.config.run_id)
        #needed when loading from a checkpoint
        # for opt in self.trainer.optimizers:
        #     for pg in opt.param_groups:
        #         pg["lr"] = self.config.lr
    
    def on_train_epoch_start(self) -> None:
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            print("starting epoch", self.current_epoch)
            
    def training_step(self, batch, batch_idx):
   
        preds = self(batch)
        preds = mymodel.gumbel_sinkhorn(preds)
        
        loss, curr_loss_components = self.loss_fn_wrapper(
            preds, batch["target_mat"], batch["n_objects"])

        batch_size = batch["n_objects"].shape[0]
        self.log(
            "train_loss", 
            loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        for k, v in curr_loss_components.items():
            self.log(
            k, v
            , on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
            
        
        # Accumulate components for logging
        # cpu_components = {k: v.item()*batch_size for k, v in curr_loss_components.items()}
        # self.epoch_loss_components += Counter(cpu_components)
        
        

        if self.config.debug_run > 0 or (batch_idx)%10==0:
            # loss_all_gpus = self.all_gather(loss.detach())
            # global_avg_loss = loss_all_gpus.mean()
            total_batches = self.trainer.num_training_batches
            #note that this is printed across every rank
            print(f"    Batch {batch_idx + 1}/{total_batches}")
            
        
        return loss

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        curr_epoch = self.current_epoch

        
        avg_train_loss = self.trainer.callback_metrics["train_loss"].item()
        utils.log({"train_loss": avg_train_loss}, step=curr_epoch)

        loss_components = {k: v for k, v in self.trainer.callback_metrics.items() if k in self.config.loss_funcs.keys()}
        utils.log(loss_components, step=curr_epoch)
        
        
        print(
            f"Epoch [{curr_epoch+1}/{self.config.max_epochs}]"
            f" Train Loss: {avg_train_loss:.6f}",
            *[f"{x}: {y:.6f}" for x, y in loss_components.items()],
            sep=" | "
        )
        if (batch := self._get_example_data()) is not None:
            print("computing output vis train")
            self.eval()
            with torch.no_grad():
                preds = self(batch)
                cl = metrics_and_losses.correct_connections(batch, preds).mean()
                utils.log({"train_connection metric (example batch only)": cl}, step=curr_epoch)
                for i in range(0, min(len(batch["img"]), self.config.show_example_n)):
                    self._log_image_visualisations(
                    preds[i], 
                    batch["target_mat"][i], 
                    batch["seg"][i], 
                    batch["img"][i], 
                    batch["pairs"][i], batch["n_objects"][i], batch["basename"][i],
                    "train prediction")
        
                        

    def on_validation_epoch_start(self) -> None:
        
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            print("validating...")
        return super().on_validation_epoch_start()


    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        
        preds = mymodel.gumbel_sinkhorn(preds)
        
        loss = self.val_loss_func(preds, batch["target_mat"], batch["n_objects"])
        
        cl = metrics_and_losses.correct_connections(batch, preds).mean()
        

        self.log(
            "val_loss", 
            loss, 
            prog_bar=False, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=batch["img"].shape[0])
        
        self.log(
            "connection_metric", 
            cl, 
            prog_bar=False, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=batch["img"].shape[0])
        
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        if not self.trainer.is_global_zero:
            return
        
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            print(f'Val Loss: {val_loss:.4f}')
            utils.log({"val_loss": val_loss.item()}, step=self.current_epoch)
        cl = self.trainer.callback_metrics.get("connection_metric")
        if cl is not None:
            utils.log({"val_connection_metric": cl.item()}, step=self.current_epoch)
        if (batch := self._get_example_data(val=True)) is not None:
            print("computing output vis val ")
            self.model.eval()
            with torch.no_grad():
                preds = self(batch)

                for i in range(0, min(len(batch["img"]), self.config.show_example_n)):
                    self._log_image_visualisations(
                    preds[i], 
                    batch["target_mat"][i], 
                    batch["seg"][i], 
                    batch["img"][i], 
                    batch["pairs"][i], batch["n_objects"][i], batch["basename"][i],
                    "val prediction")
            
    
    def get_test_ds_names(self, index) -> str:
        if hasattr(self.trainer.test_dataloaders[index], "dataset_names"):
            tmp=self.trainer.test_dataloaders[index].dataset_names
            return "+".join(tmp)
        return ""     
    
    def on_test_start(self):

        dls = self.trainer.test_dataloaders
        assert isinstance(dls, list)

        if self.trainer.is_global_zero:
            print(f"--> Detected {len(dls)} test dataloaders:")
            print([self.get_test_ds_names(i) for i in range(len(dls))])


        self.test_metrics_per_data = [
            self.metric_template.clone() for _ in range(len(dls))
        ]
        
    

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self(batch)
        
        # amax = metrics_and_losses.one_hot(preds)
        # n = batch["n_objects"]
        # print(amax[0][:n, :n])
        batch["pred"] = preds
        self.test_metrics_per_data[dataloader_idx].update(batch, preds,
                            batch)

        total_batches = int(self.trainer.num_test_batches[dataloader_idx])
        if (batch_idx+1)%10 == 0:
            print(f"    Dataset {dataloader_idx} Test Batch {batch_idx + 1}/{total_batches}")
        
        if batch_idx ==0:
            for i in range(0, min(len(batch["img"]), self.config.show_example_n)):
                self._log_image_visualisations(
                    batch["pred"][i], 
                    batch["target_mat"][i], 
                    batch["seg"][i], 
                    batch["img"][i], 
                    batch["pairs"][i], batch["n_objects"][i], batch["basename"][i],
                    "test prediction")
        

            
    def on_test_end(self) -> None:
        for dataset_idx, collection in enumerate(self.test_metrics_per_data):
            dataset_name = self.get_test_ds_names(dataset_idx)
            table = wandb.Table(columns=[f"{dataset_name} Metric", f"{dataset_name} Value"])
            # Compute returns a dict: {'metric_name': stats, worst}
            results = collection.compute()
            for metric_name, (stats, worst) in results.items():
                [table.add_data(f"{dataset_name} | {metric_name} | {k}", v) for k, v in stats.items()]
                assert worst is not None
                self._log_image_visualisations(
                    worst["pred"], worst["target_mat"],
                    worst["seg"], worst["img"],  worst["pairs"], 
                    worst["n_objects"], worst["basename"],
                    f"test pred worst {metric_name} from {dataset_name}")
                
            utils.log({f"{dataset_name}_test_results_{self.previous_run_id}": table})
            
            collection.reset()
    

    def on_save_checkpoint(self, checkpoint):
        """
        Injects extra data into the checkpoint dict before saving.
        PL automatically handles state_dict, optimizer_states, and epoch.
        """
        #print("checkpointing...")
        
        checkpoint["run_id"] = self.config.run_id
        checkpoint["wandb_run_id"] = utils.wandb_id()
        
        #print(self.model.static_relation_bias[:2, :2])
        
    def on_load_checkpoint(self, checkpoint):
        """
        Optional: Hook to read custom data when loading.
        PL automatically loads state_dict and hyperparameters.
        """
        #run id from the checkpoint is now the previous one
        self.previous_run_id = checkpoint.get("run_id")
    
    def _log_image_visualisations(self, preds, target_mat, seg, img, pairs, n, name, title: str):
        bestM = metrics_and_losses.get_optimal_assignment(preds, n)
        img_vis = metrics_and_losses.reconstruct_image(
            pairs, bestM, seg, n)
        img_vis_gt = metrics_and_losses.reconstruct_image(
            pairs, target_mat, seg, n)
        img_pair_vis = metrics_and_losses.construct_pair_img(bestM,  seg,  n)
        img_pair_vis_gt = metrics_and_losses.construct_pair_img(target_mat,  seg,  n)
        utils.log({
                f'{title} ({name})': [
                    wandb.Image(utils.to_img(img), caption="Input image"),
                    wandb.Image(img_vis, caption="Predict"),
                    wandb.Image(img_vis_gt, caption="Ground Truth"),
                    wandb.Image(img_pair_vis, caption="Pairs Predicted"),
                    wandb.Image(img_pair_vis_gt, caption="Pairs Ground Truth"),
                    
                ]
        }, step=self.current_epoch)
        




class JsonSidecarCallback(Callback):
    """
    Saves a JSON file with metadata the first time a checkpoint is written.
    """
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if checkpoint["epoch"] > 0:
            return
        # Get path of the checkpoint currently being saved
        # Note: formatting might differ slightly depending on PL version
        monitor_candidate = trainer.checkpoint_callback.best_model_path
        if not monitor_candidate: 
            return
            
        cfg: RunConfig =  pl_module.config
        json_data = {
            "run_id": cfg.run_id,
            "previous_run_id": pl_module.previous_run_id,
            "wandb_run_id": utils.wandb_id(),
            "config": vars(cfg)
        }
        # Save .json sidecar (e.g., checkpoint.ckpt -> checkpoint.json)
        ckpt_path = Path(monitor_candidate)
        json_path = ckpt_path.parent / "info.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=1)
            

def get_checkpoint_callbacks(model_name, run_id):
    root = os.environ.get("CHECKPOINT_DIR")
    if not root:
        raise EnvironmentError("CHECKPOINT_DIR not set")

    # Define path: $CHECKPOINT_DIR/<ModelClassName>/<run_id>/
    dirpath = Path(root) / model_name / run_id
    
    os.makedirs(str(dirpath), exist_ok=True)
    

    
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=f"checkpoint-{{epoch:03d}}-{{val_loss:.4f}}", # Added epoch so filename is unique per save if needed
        save_top_k=1,         # Keeps only the best model (Overwrites old ones)
        monitor="val_loss",   # Monitors validation loss
        mode="min",           # Min loss is best
        save_weights_only=False,
    )
    
    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="auto-{epoch:03d}-latest", # This will be the file name
        save_top_k=1,                 # Keep only the 1 latest
        monitor=None,                 # <--- Monitor None relies on epoch number
        every_n_epochs=1,             # Ensure it runs every epoch
        save_weights_only=False
    )

    return [checkpoint_callback, latest_checkpoint_callback, JsonSidecarCallback()]

def get_latest_ckpt_path(model_name: str, best=False) -> Path:
    root = os.environ["CHECKPOINT_DIR"]
    model_dir = Path(root) / model_name
    
    if not model_dir.exists():
        raise ValueError("Model directory not found")

    # Get last run directory
    subfolders = sorted([p for p in model_dir.iterdir() if p.is_dir()])
    if not subfolders:
        raise ValueError("No run directories found")
        
    latest_folder = subfolders[-1]
    
    # Find .ckpt files
    ckpt_files = list(latest_folder.glob("*.ckpt"))
    if not ckpt_files:
        raise ValueError(f"No .ckpt found in {latest_folder}")
    
    tmp = [x for x in ckpt_files if x.name.endswith("latest.ckpt")]
    if tmp:
        if not best:
            return tmp[0]
        ckpt_files.remove(tmp[0])
    return max(ckpt_files, key=lambda p: p.stat().st_mtime)

def load_lightning_model(config: RunConfig) ->tuple[LitModelWrapper, str | None, str | None]:
    """
    Loads the LightningModule directly.
    """
    ckpt_path = None
    # 1. Determine Path
    if config.load_from:
        ckpt_path = Path(config.load_from)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    elif config.latest or config.latest_best:
        ckpt_path = get_latest_ckpt_path(config.model, best=config.latest_best)
        
    
    

    
    model_cls = utils.get_nn_module_subclasses(mymodel)[config.model]
    model: nn.Module = model_cls(**config.model_config)
    
    if config.resume and ckpt_path is not None:
        json_path = ckpt_path.parent / "info.json"
        with open(json_path, "r") as f:
            d = json.load(f)
            assert d["config"]["datasets"] == config.datasets
            assert d["config"]["loss_funcs"] == config.loss_funcs
            assert d["config"]["batch_size"] == config.batch_size
            assert d["config"]["debug_run"] == config.debug_run
            assert d["config"]["val_split"] == config.val_split
            previous_run_id = d["run_id"]
        return LitModelWrapper(model=model, config=config), str(ckpt_path), previous_run_id
        

    # 2. Case: Start Fresh
    if ckpt_path == None:
        print("Initializing fresh model...")        
        return LitModelWrapper(model=model, config=config),None, None

    # 3. Case: Load from Checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    
    
    
    # LitModel.load_from_checkpoint will:
    # 1. Read 'hyper_parameters' from the checkpoint
    # 2. Call LitModel.__init__(**hyper_parameters) to create the object
    # 3. Load state_dict
    # 4. Call on_load_checkpoint (where we read wandb_id etc)
    # Therefore, all set attributes will be present
    
    #needed because it's not a primitive type
    with torch.serialization.safe_globals([RunConfig]):
        lit_model = LitModelWrapper.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, config = config)
        raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    
    assert hasattr(lit_model, "config")
    
    # current_bias = lit_model.model.static_relation_bias.detach().clone().cpu()
    # diff = (current_bias - initial_bias).abs().sum().item()
    # print(f"Total weight change since start: {diff}")
    # print(current_bias[:2,:2])
    # print(initial_bias[:2, :2])



    return lit_model, None, raw_ckpt.get("run_id")





