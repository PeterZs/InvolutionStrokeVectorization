# Intersection model

Stage 3 of the pipeline. This subproject trains the *intersection model*, which decides which half edges
(called half-branches in the paper) belong to the same stroke.

## What it predicts

For each drawing, the model receives the polyline (point coordinates) of every half edge, and a constraint matrix `R` that marks which pairs of half edges are
allowed to connect (they have to meet at the same junction). It outputs a soft
assignment matrix over the half edges, normalized with Sinkhorn iterations so that the
result is close to a permutation. Each entry indicates the likelihood that two half edges connect.

> [!NOTE]
> We previously experimented with image-based representations, where the model takes an image and a segmentation mask as input instead. This approach is deprecated.

## Implementation overview

Two models are defined in `mymodel.py`, both sharing `SceneGraphModelBase`:

- `SceneGraphImageModel`: an image backbone that pools per-object features from the
  segmentation mask. Deprecated.
- `SceneGraphVectorModel`: a polyline backbone that consumes the half edge geometry
  directly. This is the variant exported for the pipeline.

Both feed a `RelationSetTransformer` and a `SymmetricOutputHead`. The Gumbel-Sinkhorn
normalization is implemented in `mymodel.py`, `permutations.py` contains permutation utilities, and the
corresponding tests are under `test/`.
Training is built on PyTorch Lightning (`litmodel.py`), with losses and metrics in
`metrics_and_losses.py`.

## Setup

```bash
uv sync
```

Configuration is read from `config.env`:

```env
DATA_ROOT=<path/to/extracted/dataset>
PERMANENT_DATA_PATH=<path/to/dataset/archives>
CHECKPOINT_DIR=checkpoints
WANDB_PROJECT=intersection_model
WANDB_CONSOLE=off
```

Leave `WANDB_PROJECT` empty to disable logging. `CHECKPOINT_DIR` is where checkpoints are saved.

`DATA_ROOT` points to the dataset produced by [`intersection_dataset`](../intersection_dataset/README.md), with
`train/<DatasetName>` and `test/<DatasetName>` subfolders. `PERMANENT_DATA_PATH` contains the dataset archives.
In our workflow, the dataset is extracted to `DATA_ROOT` for each run. This is done by the `scripts/prepare_data_parallel.py` script,
which you do not need to call manually, because the SLURM script handles it.

## Workflow

The entry point is `main.py`, with a required `--mode`: one of `train`, `test`,
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

`--latest-best` loads the best checkpoint of the most recent run, `--latest` loads the
most recent checkpoint, and `--load_from <path>` loads a specific checkpoint.

### Trace (export for the pipeline)

Export a TorchScript model. `--trace_device` controls which device the export targets,
which matters because the `main/` pipeline runs on the CPU or MPS:

```bash
uv run main.py --model SceneGraphVectorModel --latest-best --mode trace \
  --trace_device cpu --save_to output_cpu.pt
```

Alternatively, `trace_export.py` traces a given checkpoint file directly:

```bash
uv run trace_export.py --model SceneGraphVectorModel --save_to output_cpu.pt --trace_device cpu <checkpoint>
```

Point `INTERSECTION_MODEL` in `main/config.env` to the resulting file. You can sanity-check
an export against its checkpoint:

```bash
uv run test/test_exported_model.py <checkpoint.ckpt> output_cpu.pt
```

## Pre-trained weights

Checkpoints and TorchScript exports are not tracked in git. You can train and trace
your own model, or download the pre-trained `SceneGraphVectorModel` checkpoint and trace it:

```bash
curl -LO https://igl.ethz.ch/projects/involution-stroke-vectorization/intersection_checkpoint.pt
uv run trace_export.py --model SceneGraphVectorModel --save_to intersection_traced.pt --trace_device cpu intersection_checkpoint.pt
```

## Notes

- A batch size of 16 with bf16 fits in memory for the standard configuration.
- The maximum number of objects and points are model hyperparameters set in `RunConfig`
  (`model_config`).
