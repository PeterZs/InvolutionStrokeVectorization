# Dataset generation (centerline pairs)

Generates the training image pairs for the [centerline model](../centerline_model/README.md).

We use a mix of synthetic data and vector drawing datasets found in the wild, all in stroke or
outline form. Each source dataset has its own loader, because each one needs different
preprocessing to be normalized into a common stroke representation.

## How a pair is made

For every source drawing, `drawutils.generate_image_pairs` renders the strokes with our
brush engine to produce the input, and renders a thin centerline to produce the target.
Each drawing is rendered several times with different brushes and parameter variations,
so one vector source yields several training pairs.

## Brush engine

`brushengine/` is a custom brush engine that imitates, to a degree, the brush mechanics
of tools like Photoshop. The base brushes are defined in `brushengine/brushes`, and
`BRUSHES_TO_USE` in `process_everything.py` selects the ones used for generation.
`brush_variations.json` defines small parameter perturbations applied per render. Some
brushes need downscaling or upscaling to look right; this is handled in `process_everything.py`.
Use `brush_display.py` to preview the available brushes.

## Layout

| Path                    | Content                                                                                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `process_everything.py` | The main generation driver.                                                                                                                                                         |
| `drawutils.py`          | Rendering of input and target pairs.                                                                                                                                                |
| `brushengine/`          | The brush engine and base brushes.                                                                                                                                                  |
| `dataset_loaders/`      | One loader per source dataset (TU Berlin, QuickDraw, Synthetic, GeCreative, PimpMyDrawing, instance segmentation). `loader_utils.find_loader_classes` discovers them by class name. |
| `utils.py`              | Helper code. SVG helpers live in the shared [`svgutils`](../svgutils/README.md) package.                                                                                            |
| `make_eval_set.py`      | Builds a held-out evaluation set.                                                                                                                                                   |
| `scripts/`              | Helper scripts, e.g., one submission script per dataset for the cluster.                                                                                                            |

## Setup

```bash
uv sync
```

## Source datasets

We use several source vector datasets, which need to be downloaded separately (around 18 GB in total).
See [Obtaining the datasets](#obtaining-the-datasets) for download instructions. Alternatively, you can create your own dataset and loaders.
`SyntheticDataset` is generated procedurally and requires no download.

The dataset loaders expect all datasets to be in a single folder, whose location is specified in a `config.env` file that you need to create.
The loaders append the dataset folder names directly to this path, so it must end with a `/`:

```env
dataset_root=your/path/datasets/
```

That folder must be structured as follows:

```
datasets
├── gecreative
│   └── processed_data
│       ├── bird_short_beak_json_64
│       ├── bird_short_body_json_64
│       └── ... 30 more
├── instancesegmentation
│   ├── clipasso-base_train
│   │   ├── DRAWING_GT_SVG
│   │   └── ...
│   └── sketchagent_train
│       ├── DRAWING_GT_SVG
│       └── ...
├── quickdraw
│   ├── aircraft carrier.npz
│   ├── airplane.npz
│   └── ... 343 more
└── TU berlin Sketch Dataset
    └── svg
        ├── airplane
        ├── alarm clock
        └── ... 250 more
```

## Generating

The entry point is `process_everything.py`. Select which source datasets to render by
loader class name (e.g., `QuickDrawDataset`), or omit `--class` to render all of them.

```bash
# a small selection across all datasets
uv run process_everything.py --out_dir dataset_sample/output \
  --num_brushes_per_sample 5 --offset 100 --to 101

# a fixed range from one dataset
uv run process_everything.py --out_dir datasets/OUT_TUBERLIN/1_1000 \
  --class TUBerlinDataset --offset 0 --to 1000
```

Mixing all datasets (by omitting `--class`) should be reserved for debugging.
In the end, we want a distinct collection for each source dataset, to have more control during training.

| Flag                       | Meaning                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------- |
| `--out_dir`                | Output directory. A `filelist.jsonl` log is appended there.                        |
| `--class`                  | One or more loader class names. Empty means all datasets.                          |
| `--split`                  | Dataset split to draw from (default `train`).                                      |
| `--target_size`            | Output image size (default 1000).                                                  |
| `--offset`, `--to`         | Inclusive index range of source samples to render.                                 |
| `--num_brushes_per_sample` | Render with this many brushes per sample, chosen at random. `-1` uses all of them. |
| `--skip_variations`        | Do not apply the brush parameter variations.                                       |

Generation is resumable: it counts how many samples already exist for a class and
continues from there (overwriting the latest one found, for consistency in case of a crash).

### On a cluster

Large runs are submitted to a SLURM cluster through the per-dataset
scripts under `scripts/` (e.g., `scripts/submitTUBerlinDataset.sh`). In that case, `config.env` must also define `OUT_DIR` (the output location) and `SLURM_ACCOUNT`.
To benefit from parallelism, samples are sharded, with 100 source samples assigned to each shard. The output looks as follows:

```
OUT_DIR
└── GeCreativeDataset
    ├── shard_0
    ├── shard_100
    └── ... 198 more
```

## Preparing the data for training

To make the generated datasets ready for training, each one needs to be archived into a tar file, and a CSV file mapping inputs to outputs must be created.

### Creating pairs

Locally:

```bash
uv run scripts/create_pairs.py --root datasets/OUTPUT/raw_data/test/QuickDrawDataset
```

On a cluster:

```bash
srun uv run python -u scripts/create_pairs.py --root output
```

### Creating the tar archive

All shards are consolidated into a single tar file. File paths in the archive are rewritten so that the shard folder is stripped away: `shard_0/image.webp` → `image.webp`.

```bash
bash scripts/run_python.sh scripts/make_tar.py --base_dir youroutputdir/TUBerlinDataset --tar_dir youroutputdir/TUBerlinDataset.tar
```

## Obtaining the datasets

|     | Paper/Name                                                         | Number of samples  | Paper link                                             | Dataset link                                                                                                       |
| --- | ------------------------------------------------------------------ | ------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| 1   | Instance Segmentation of Scene Sketches Using Natural Image Priors | 11,234 (5,617 × 2) | [link](https://inklayer.github.io/)                    | [link](https://www.inkscenes-dataset.com/download)                                                                 |
| 2   | TU Berlin: How Do Humans Sketch Objects? (2012)                    | 20,000             | [link](https://dl.acm.org/doi/10.1145/2185520.2185540) | [link](https://web.archive.org/web/20250131215015/https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) |
| 3   | Sketch-RNN QuickDraw Dataset                                       | 24,150,000         | -                                                      | [link](https://github.com/googlecreativelab/quickdraw-dataset)                                                     |
| 4   | Ge et al. 2021: Creative Sketch Generation                         | 167,023            | [link](https://arxiv.org/abs/2011.10039)               | [link](https://songweige.github.io/projects/creative_sketech_generation/gallery_creatures.html)                    |

To obtain the QuickDraw dataset, you need the `gsutil` tool. Then proceed as follows:

```bash
mkdir quickdraw
gsutil ls gs://quickdraw_dataset/sketchrnn/*.npz | grep -v '\.full\.npz$' | gsutil -m cp -I quickdraw
```
