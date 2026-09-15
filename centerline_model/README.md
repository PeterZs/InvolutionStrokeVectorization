# Centerline model

This subproject trains the U-Net that maps a rendered line drawing to its centerline
(stage 1 of the pipeline). The trained model is exported to TorchScript and loaded
by the [`main/`](../README.md#running-the-vectorization-pipeline) pipeline.

## Models

Two architectures are defined in `mymodel.py`:

- `ResNextUNet`: a ResNeXt encoder with a U-Net decoder.
- `ResNextUNetLarge`: a wider variant, used for the final results.

Training uses a custom loop (`train.py`) with Weights & Biases logging. Losses and
metrics are defined in `metrics_and_losses.py`.

## Setup and data workflow

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

Leave `WANDB_PROJECT` empty to disable wandb logging. `CHECKPOINT_DIR` is where checkpoints are saved.

Training includes a preprocessing step: the archived datasets are copied and unpacked to a suitable location.
This setup accommodates the faster but limited storage typically found on clusters.
`DATA_PATH` points to the archives and pair CSVs (one of each per dataset) produced by [`dataset_gen`](../dataset_gen/README.md):

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

The `scripts/prepare_data.py` script copies and unpacks the data to `SCRATCH_DIR`, from which training and testing actually read.
It usually does not need to be called manually, as the SLURM submission script calls it. `SCRATCH_DIR` looks as follows:

```
SCRATCH_DIR
└── train
    ├── GeCreativeDataset
    │   ├── image.webp
    │   └── ... 100000 more
    ├── GeCreativeDataset.csv
    └── ... more
```

## Workflow

The entry point is `main.py`, with a required `--mode`. The four modes are `train`,
`test`, `predict` and `trace`. By default, a CUDA device is required; pass `--cpu` to
run on the CPU.

### Train

On a SLURM cluster, training is submitted through `scripts/submit2.sh`. Training is distributed over 8 GPUs on a single node.
You can modify the script to adapt it to your SLURM configuration.
A submission looks as follows:

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

`--loss_funcs` takes a comma-separated list of `LossName:weight` pairs.

Local debug run on the CPU (make sure the unpacked data is available in `SCRATCH_DIR`):

```bash
uv run main.py --model ResNextUNet --mode train --datasets TUBerlinDatasetSmall \
  --debug_run 10 --cpu --batch_size 1 --val_split 0.5
```

### Test

```bash
uv run main.py --model ResNextUNetLarge --mode test --datasets testset --batch_size 1 --cpu --latest
```

Pass `--latest` to load the most recent checkpoint, or `--checkpoint <path>` to load a
specific one.

> [!NOTE]
> Running the test set has not been tested on the cluster.

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

Point `CENTERLINE_MODEL` in `main/config.env` to the resulting file.

## Pre-trained weights

Checkpoints and TorchScript exports are not tracked in git. You can train and trace
your own model, or download the pre-trained `ResNextUNetLarge` checkpoint and trace it:

```bash
curl -LO https://igl.ethz.ch/projects/involution-stroke-vectorization/centerline_checkpoint.pt
uv run trace_export.py ResNextUNetLarge centerline_checkpoint.pt centerline_traced.pt
```
