import wandb
import torch
from torch.utils.data import DataLoader
from collections import Counter
from metrics_and_losses import build_loss_func, masked_l2_loss, masked_median_area, soft_dice_loss
import metrics_and_losses
import utils
import metrics_and_losses
from contextlib import nullcontext
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback
from runconfig import RunConfig
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection

class LitModel(pl.LightningModule):
    def __init__(self, model, config: RunConfig, previous_state_dict):
        super().__init__()
        self.model = model
        self.config = config
        self.previous_state_dict = previous_state_dict
        self.loss_fn_wrapper = None
        
        # State for manual accumulation to match original logging format
        self.epoch_loss_components = Counter()
        self.epoch_example_data = None
        self.val_example_data = None
        self.best_val_loss = float('inf')
        self.val_loss_func = torch.nn.MSELoss()
        self.metric_template = MetricCollection({
            'masked_l2_loss': metrics_and_losses.GlobalStatsMetric(masked_l2_loss),
            'masked_median_area':metrics_and_losses.GlobalStatsMetric(masked_median_area),
            'soft_dice_loss':metrics_and_losses.GlobalStatsMetric(soft_dice_loss)
        })

    def setup(self, stage: str):
        self.loss_fn_wrapper = build_loss_func(self.config.loss_funcs, self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        if self.previous_state_dict.get("optimizer_state") is not None:
            optimizer.load_state_dict(self.previous_state_dict["optimizer_state"])
            # Reset LR just in case
            for g in optimizer.param_groups:
                g['lr'] = self.config.lr
        
        return optimizer
    
    def get_test_ds_names(self, index) -> str:
        if hasattr(self.trainer.test_dataloaders[index], "dataset_names"):
            tmp=self.trainer.test_dataloaders[index].dataset_names
            if len(tmp)==1:
                return tmp[0]
            return "+".join(tmp)
        return ""


    def on_train_start(self):
        if self.epoch_example_data is None:
            print("get examples for output show")
            train_loader = self.trainer.train_dataloader
            self.epoch_example_data = next(iter(train_loader))
            val_loader = self.trainer.val_dataloaders
            self.val_example_data = next(iter(val_loader))

    def on_train_epoch_start(self):
        self.epoch_loss_components = Counter()

    def training_step(self, batch, batch_idx):
        input_imgs, output_imgs, masks, names = batch

        preds = self.model(input_imgs)
        if self.config.masking: 
            preds = preds * masks
            output_imgs = output_imgs * masks
            
        loss, curr_loss_components = self.loss_fn_wrapper(preds, output_imgs)

        # Accumulate components for logging
        cpu_components = {k: v.item() for k, v in curr_loss_components.items()}
        self.epoch_loss_components += Counter(cpu_components)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=input_imgs.shape[0], sync_dist=True)

        # Get total batches (this works for standard sized dataloaders)
        if (batch_idx+1)%10 == 0:
            loss_all_gpus = self.all_gather(loss.detach())
            global_avg_loss = loss_all_gpus.mean()
            total_batches = self.trainer.num_training_batches
            print(f"    Batch {batch_idx + 1}/{total_batches} Loss: {global_avg_loss:.4f}")
        return loss

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        epoch = self.current_epoch
        
        # Compute averages
        avg_train_loss = self.trainer.callback_metrics["train_loss"].item()
            
        num_batches = self.trainer.num_training_batches
        avg_components = {k: v / num_batches for k, v in self.epoch_loss_components.items()}

        utils.log({"train_loss": avg_train_loss}, step=epoch)
        utils.log(avg_components, step=epoch)
        if utils.get_rank()==0:
            #here only the local losses are printed because they're never shared
            print(
                f"Epoch [{epoch}/{self.config.num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f}",
                *[f"{x}: {y:.4f}" for x, y in avg_components.items()],
                sep=" | "
            )

        # Show Output Every N epochs
        if self.config.show_output_every >= 0 and epoch % self.config.show_output_every == 0:
            self.eval()
            with torch.no_grad():
                inputs_example, target_example, masks_example, name_example = self.epoch_example_data
                input_moved = inputs_example.to(self.device)
                preds = self.model(input_moved)
                for nm, pr, tr, inp, msk in zip(
                    name_example, preds, target_example, inputs_example, masks_example):
                    utils.log({
                        f"prediction_example ({nm})": [
                            wandb.Image(utils.to_img(pr), caption="Predicted"),
                            wandb.Image(utils.to_img(tr), caption="Ground Truth"),
                            wandb.Image(utils.to_img(inp), caption="Input"),
                            wandb.Image(utils.to_img(msk), caption="Mask"),
                        ]
                    }, step=epoch)
            self.train()

    def validation_step(self, batch, batch_idx):
        input_imgs, output_imgs, masks, name = batch
        preds = self.model(input_imgs)
        
        if self.config.masking: 
            preds = preds * masks
            output_imgs = output_imgs * masks
            
        loss = self.val_loss_func(preds, output_imgs)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, sync_dist=True, batch_size=input_imgs.shape[0])

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        if not self.trainer.is_global_zero:
            return
        
       
        self.eval()
        with torch.no_grad():
            inputs_example, target_example, masks_example, name_example = self.val_example_data
            input_moved = inputs_example.to(self.device)
            preds = self.model(input_moved)
            for nm, pr, tr, inp, msk in zip(
                name_example, preds, target_example, inputs_example, masks_example):
                utils.log({
                    f"prediction_example ({nm})": [
                        wandb.Image(utils.to_img(pr), caption="Predicted"),
                        wandb.Image(utils.to_img(tr), caption="Ground Truth"),
                        wandb.Image(utils.to_img(inp), caption="Input"),
                        wandb.Image(utils.to_img(msk), caption="Mask"),
                    ]
                }, step=self.current_epoch)
            
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            print(f'Val Loss: {val_loss:.4f}')
            utils.log({"val_loss": val_loss.item()}, step=self.current_epoch)
        
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            current_epoch = self.trainer.current_epoch
            
            utils.save_model_checkpoint(
                self.model, 
                self.optimizers(), 
                self.config.run_id, 
                current_epoch, 
                val_loss.item(), 
                overwrite=True,
                previous_run_id=self.previous_state_dict["run_id"], 
                extra_data=vars(self.config)
            )
            print(current_epoch, f"saved best checkpoint (val_loss: {val_loss:.4f})")

    def on_test_start(self):
        """
        Called immediately before the test loop begins.
        The trainer now has the dataloaders attached.
        """
        # self.trainer.test_dataloaders is usually a list. 
        # If a single loader is passed, PL might wrap it or leave it singular depending on version.
        dls = self.trainer.test_dataloaders
        if not isinstance(dls, list):
            dls = [dls]
        
        num_dataloaders = len(dls)
        
        print(f"--> Detected {num_dataloaders} test dataloaders. Initializing metrics...")

        # 3. Create a ModuleList of MetricCollections (one per dataloader)
        # We clone the template so they maintain separate states.
        self.test_metrics_per_data = [
            self.metric_template.clone() for _ in range(num_dataloaders)
        ]
        
        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input_imgs, output_imgs, masks, names = batch
        preds = self.model(input_imgs)
        

        self.test_metrics_per_data[dataloader_idx].update((preds, output_imgs, masks),
                            {"inp": input_imgs, "pred":preds, "gt": output_imgs, "name": names})

        total_batches = int(self.trainer.num_test_batches[dataloader_idx])
        if (batch_idx+1)%10 == 0:
            print(f"    Dataset {dataloader_idx} Test Batch {batch_idx + 1}/{total_batches}")
        
        if self.trainer.is_global_zero and batch_idx%(total_batches//10) == 0:
            print("uploading eval example predictions to wandb")
            #for i in range(input_imgs.shape[0]):
            utils.log({
                f"{self.get_test_ds_names(dataloader_idx)} test_prediction_example_{names[0]}": [
                    wandb.Image(utils.to_img(preds[0]), caption="Predicted"),
                    wandb.Image(utils.to_img(output_imgs[0]), caption="Ground Truth"),
                    wandb.Image(utils.to_img(input_imgs[0]), caption="Input"),
                ]
            })

    
    def on_test_end(self) -> None:
        for dataset_idx, collection in enumerate(self.test_metrics_per_data):
            dataset_name = self.get_test_ds_names(dataset_idx)
            table = wandb.Table(columns=[f"{dataset_name} Metric", f"{dataset_name} Value"])
            # Compute returns a dict: {'acc': val, 'prec': val}
            results = collection.compute()
            for metric_name, (stats, worst) in results.items():
                
                
                [table.add_data(f"{dataset_name} | {k}", v) for k, v in stats.items()]
                
                pred, output_img, input_img = worst["pred"], worst["gt"], worst["inp"]
                utils.log({
                        f"{dataset_name} {metric_name} test_prediction_worst_{worst['name']}": [
                            wandb.Image(utils.to_img(pred), caption="Predicted"),
                            wandb.Image(utils.to_img(output_img), caption="Ground Truth"),
                            wandb.Image(utils.to_img(input_img), caption="Input"),
                        ]
                })
                
            utils.log({f"{dataset_name}_test_results_{self.previous_state_dict['run_id']}": table})
            
            collection.reset()

    def forward(self,  input_imgs):
        preds = self.model(input_imgs)
        return preds



def train_model(train_loader_batched, 
                val_loader, 
                model: torch.nn.Module, 
                config: RunConfig, 
                previous_state_dict: dict):
    
    print("starting training")
    
    lit_model = LitModel(model, config, previous_state_dict)
    precision = "bf16-mixed" if config.use_bf16 else 32
    
    # 1. Early Stopping Callback
    callbacks = []
    if val_loader and config.patience_early_stop >0:
        callbacks.append( EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=config.patience_early_stop,
            verbose=True,
            mode="min"
        )
        )

    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',  
        num_nodes=1,
        strategy='auto' if not torch.cuda.is_available() else 'ddp',
        use_distributed_sampler=True,
        max_epochs=config.num_epochs,
        precision=precision,
        enable_checkpointing=False, # Disable default to use our custom callback
        logger=False,
        # validate every epoch is now the default/forced behavior
        check_val_every_n_epoch=1 if val_loader else None,
        callbacks=callbacks,
        enable_progress_bar=False,
    )
    print(trainer.strategy)

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader_batched,
        val_dataloaders=val_loader
    )
    
def test_model(test_loaders: list, 
                model: torch.nn.Module, 
                config: RunConfig, 
                previous_state_dict: dict):
    
    lit_model = LitModel(model, config, previous_state_dict)
    
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
    

def trace_for_export(out_path, model: torch.nn.Module, config: RunConfig, previous_state_dict):
    lit_model = LitModel(model, config, previous_state_dict)
    lit_model.eval()
    #image size doesn't really matter since it's fully convolutional
    randinp = torch.randn(1, 1, config.image_size, config.image_size)
    print("torchscript tracing...")
    script = lit_model.to_torchscript(
        method="trace", example_inputs=randinp)
    torch.jit.save(script, out_path)
