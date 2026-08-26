# Centerline model

This subproject trains the U-Net that maps a rendered line
drawing to its centerline. The trained model is exported to TorchScript and loaded
by the `main/` pipeline.

## Models

Two architectures are defined in `mymodel.py`:

- `ResNextUNet`: a ResNeXt encoder with a U-Net decoder.
- `ResNextUNetLarge`: a wider variant used for the final results.

Training uses a custom loop (`train.py`) with Weights & Biases logging. Losses and
metrics live in `metrics_and_losses.py`.

## Setup & Data Workflow

```bash
uv sync
```

Configuration is read from `config.env`:

```env
SCRATCH_DIR=<path/to/prepared/data>
DATA_PATH=../dataset_gen/datasets/OUTPUT
CHECKPOINT_DIR=checkpoints
WANDB_PROJECT=centerline_model
WANDB_CONSOLE=off
DEBUG=True
```
Set
`WANDB_PROJECT` empty to disable wandb logging. `CHECKPOINT_DIR` is where checkpoints will be saved.

Training is done with a preprocessing step: the archived datasets are copied and unpacked to a suitable location. This setup is to accomodate faster but limited storage found on clusters. 
`DATA_PATH` points to the archives + pair csvs (one for each dataset) produced by `dataset_gen`:
```
DATA_PATH
└── train
    ├── GeCreativeDataset.csv
    ├── GeCreativeDataset.tar
    ├── QuickDrawDataset.csv
    ├── QuickDrawDataset.tar
    ├── SyntheticDataset.csv
    ├── SyntheticDataset.tar
    ├── TUBerlinDataset.csv
    └── TUBerlinDataset.tar
```
The  `prepare_data.py` script writes and unpacks to `SCRATCH_DIR`, where training and testing actually read. This script usually doesn't need to be called manually, as it is called from the SLURM submission script. `SCRATCH_DIR` would look as follows:
```
└── train
    ├── GeCreativeDataset
    │   ├── image.webp
    │   └── ... 100000 more
    ├── GeCreativeDataset.csv
    └── ... more
```



## Workflow

The entry point is `main.py` with a required `--mode`. The four modes are `train`,
`test`, `predict` and `trace`. By default a CUDA device is required; pass `--cpu` to
run on CPU.

### Train


On a SLURM cluster, training is submitted through `scripts/submit2.sh`. Training is distributed on 8 GPUs on one node. You can modify the script to adapt it to your SLURM configuration. 
A submission would look as follows:

```bash
sbatch scripts/submit2.sh main.py --model ResNextUNet --mode train \
  --datasets TUBerlinDataset QuickDrawDataset SyntheticDataset \
  --batch_size 14 --num_epochs 100 --use_bf16
```

The large model with custom loss weights:

```bash
sbatch scripts/submit2.sh main.py --model ResNextUNetLarge --mode train \
  --datasets TUBerlinDataset QuickDrawDataset SyntheticDataset GeCreativeDataset \
  --batch_size 8 --num_epochs 100 --use_bf16 \
  --loss_funcs MSELoss:1.0,soft_dice_loss:0.001
```

`--loss_funcs` takes a comma separated list of `LossName:weight` pairs.

Local debug run on CPU:

```bash
uv run main.py --model ResNextUNet --mode train --datasets TUBerlinDatasetSmall \
  --debug_run 10 --cpu --batch_size 1 --val_split 0.5
```
Make sure that the unpacked data is available in `SCRATCH_DIR`. 

### Test

```bash
uv run main.py --model ResNextUNetLarge --mode test --datasets testset --batch_size 1 --cpu --latest
```

Pass `--latest` to load the most recent checkpoint, or `--checkpoint <path>` for a
specific one.
Note: running the test set has not been tested on the cluster.

### Predict

Run a selected model on individual images:

```bash
uv run main.py --model ResNextUNet --mode predict --images bike.png --latest
```

### Trace (export for the pipeline)

Export a TorchScript model for use by `main/`:

```bash
uv run trace_export.py ResNextUNetLarge "checkpoints/yourcheckpoint.pt" output.pt
```

Point `CENTERLINE_MODEL` in `main/config.env` at the resulting file.

## Pre-trained weights

Checkpoints and TorchScript exports are not tracked in git. You can train and trace
your own, or download the pre-trained weights: TODO add download link.

