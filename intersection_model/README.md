# Intersection model

Stage 3 of the pipeline. This folder is concerned with training the *intersection model*, which decides which half edges belong to
the same stroke. 
## What it predicts

For each drawing the model receives the polyline (point coordinates) of every half-edge, and a constraint matrix `R` that marks which half edge pairs are even
allowed to connect (they have to meet at the same junction). It outputs a soft
assignment matrix over the half edges, normalized with Sinkhorn iterations so that the
result is close to a permutation. An entry indicates the likelyhood that two half-edges connect.

Note: previously we experimented with image-based representations where the model would take an image and segmentation mask instead. This is deprecated.

## Implementation Overview

Defined in `mymodel.py`, both sharing `SceneGraphModelBase`:

- `SceneGraphImageModel`: an image backbone that pools per object features from the
  segmentation mask. Deprecated.
- `SceneGraphVectorModel`: a polyline backbone that consumes the half edge geometry
  directly. This is the variant exported for the pipeline.

Both feed a `RelationSetTransformer` and a `SymmetricOutputHead`. The Sinkhorn
normalization lives in `permutations.py`, with correctness tests under `test/`.
Training is built on PyTorch Lightning (`litmodel.py`), with losses and metrics in
`metrics_and_losses.py`.

## Setup

```bash
uv sync
```

Configuration is read from `config.env`:

```env
DATA_ROOT=../intersection_dataset/debug_files
PERMANENT_DATA_PATH=../intersection_dataset/debug_files/permanent
CHECKPOINT_DIR=checkpoints
WANDB_PROJECT=intersection_model
WANDB_CONSOLE=off
```

Set `WANDB_PROJECT` empty
to disable logging. `CHECKPOINT_DIR` is where checkpoints will be saved.

`DATA_ROOT` points at the dataset produced by `intersection_dataset`, with
`train/<DatasetName>` and `test/<DatasetName>` subfolders. `PERMANENT_DATA_PATH` contains the archive of the dataset. In our worflow, the dataset is extracted each time to `DATA_ROOT`. This is done with the `prepare_data.py` script, but you don't need to call it because it is handled in the SLURM script.


## Workflow

The entry point is `main.py` with a required `--mode`, one of `train`, `test`,
`predict` (not implemented) or `trace`. Pass `--cpu` to run without a GPU.

### Train

Local debug run for the vector model:

```bash
uv run main.py --model SceneGraphVectorModel --mode train --datasets Dataset2 \
  --debug_run 4 --cpu --batch_size 2 --val_split 0.5 --max_epochs 2 \
  --loss_funcs MaskedCrossEntropyLoss:1.0,MSELoss:1.0 --patience_early_stop 0
```

On the cluster, through `scripts/submit.sh`:

```bash
sbatch scripts/submit.sh --model SceneGraphVectorModel --mode train \
  --datasets Dataset5Archive DatasetRasterized1 --batch_size 10 --max_epochs 100 \
  --loss_funcs MaskedCrossEntropyLoss:1.0,MSELoss:1.0 --patience_early_stop 0 --use_bf16
```

Add `--resume --latest` to continue from the most recent checkpoint. 
### Test

```bash
uv run main.py --model SceneGraphVectorModel --mode test --datasets TestDataset1 \
  --cpu --batch_size 1 --latest-best
```

`--latest-best` loads the best checkpoint of the most recent run; `--latest` loads the
most recent one; `--load_from <path>` loads a specific checkpoint.

### Trace (export for the pipeline)

Export a TorchScript model. `--trace_device` controls which device the export targets,
which matters because the `main/` pipeline runs on CPU or MPS:

```bash
uv run main.py --model SceneGraphVectorModel --latest-best --mode trace \
  --trace_device cpu --save_to output_cpu.pt
```

Point `INTERSECTION_MODEL` in `main/config.env` at the resulting file. You can sanity
check an export against its checkpoint:

```bash
uv run test/test_exported_model.py <checkpoint.ckpt> output_cpu.pt
```

## Pre-trained weights

Checkpoints and TorchScript exports are not tracked in git. You can train and trace
your own, or download the pre-trained weights: TODO add download link.

## Notes

- Batch size 16 with bf16 fits in memory for the standard configuration.
- The maximum object and point counts are model hyperparameters set in `RunConfig`
  (`model_config`).
