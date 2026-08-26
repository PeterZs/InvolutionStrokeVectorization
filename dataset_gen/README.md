# Dataset generation (centerline pairs)

Generates the training image pairs for the `centerline_model`. 

We use a mix of synthetic data and vector drawing datasets found in the wild, all in stroke or
outline form. Each source dataset has its own loader because each needs different
preprocessing to be normalized into a common stroke representation.

## How a pair is made

For every source drawing, `drawutils.generate_image_pairs` renders the strokes with our
brush engine to produce the input, and renders a thin centerline to produce the target.
Each drawing is rendered several times with different brushes and parameter variations,
so one vector source gives several training pairs.

## Brush engine

`brushengine/` is a custom brush engine that imitates, to a degree, the brush mechanics
of tools like Photoshop. There are 10 base brushes in `brushengine/brushes`, and
`brush_variations.json` defines small parameter perturbations applied per render. Some
brushes need down or upscaling to look right, handled in `process_everything.py`. Use
`brush_display.py` to preview the available brushes.

## Layout

- `process_everything.py`: the main generation driver.
- `drawutils.py`: rendering of input and target pairs.
- `brushengine/`: the brush engine and base brushes.
- `dataset_loaders/`: one loader per source dataset (TU Berlin, QuickDraw, Synthetic,
  Ge Creative, PimpMyDrawing, instance segmentation). `loader_utils.find_loader_classes`
  discovers them by class name.
- `svgutils.py`, `utils.py`: SVG and helper code.
- `make_eval_set.py`: builds a held out evaluation set.
- `scripts/`: helper scripts, e.g., there's one source script per dataset for cluster submission.

## Source Datasets

We use a number of source vector datasets, which you need to download the separately, and together they are around 18 GB. 
Head over to [Obtaining the datasets](#obtaining-the-datasets) to obtain them. Otherwise you can of course create your own dataset and loaders.

 The dataset loaders assume they are all in a specific folder whose location is specified in a `config.env` file you need to create:

```
dataset_root=your/path/datasets
```
That folder must look as as follows:
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




## Setup

```bash
uv sync
```

## Generating

The entry point is `process_everything.py`. Select which source datasets to render by
class name, or omit `--class` to render all of them. The class name is identified by the name of the dataset loader class (example: `QuickDrawDataset`)

```bash
# a small selection across all datasets
uv run process_everything.py --out_dir dataset_sample/output \
  --num_brushes_per_sample 5 --offset 100 --to 101

# a fixed range from one dataset
uv run process_everything.py --out_dir datasets/OUT_TUBERLIN/1_1000 \
  --class TUBerlinDataset --offset 0 --to 1000
```

Mixing all datasets (by omitting `--class`) should only be reserved for debugging purposes.
In the end, we want to obtain a distinct collection for each source dataset, to have more control during training.

Large runs are submitted on the cluster through the per dataset
scripts under `scripts/`. In that case, you must also have `OUT_DIR` set in your `config.env`file, to specify the output location. To profit from parallelism, samples are sharded, with 100 source samples assigned to each shard. The output will look as follows:
```
OUT_DIR
└── GeCreativeDataset
    ├── shard_0
    ├── shard_100
    └── ... 198 more
```


Generation is resumable: it counts how many samples already exist for a class and
continues from there (overwriting the latest found one for consistency in case of a crash). 


| Flag | Meaning |
| ---- | ------- |
| `--out_dir` | Output directory. A `filelist.jsonl` log is appended there. |
| `--class` | One or more loader class names. Empty means all datasets. |
| `--split` | Dataset split to draw from (default `train`). |
| `--target_size` | Output image size (default 1000). |
| `--offset`, `--to` | Inclusive index range of source samples to render. |
| `--num_brushes_per_sample` | Render with this many brushes per sample, chosen at random. `-1` uses all. |
| `--skip_variations` | Do not apply the brush parameter variations. |

## Make it ready for training

To make the generated datasets ready for training, they need to be archived to a tar file, and a csv file mapping input - output pairs must be created:

#### Creating pairs
local
```
uv run scripts/create_pairs.py --root datasets/OUTPUT/raw_data/test/QuickdrawDataset
```

cluster
```
srun uv run python -u scripts/create_pairs.py --root output
```

#### tar creation
All shards are consolidated in a single tar file. The file paths in the archive are rewritten so that the shard folder is stripped away: `shard_0/image.webp` -> `image.webp`.
```
bash scripts/run_python.sh scripts/make_tar.py --base_dir youroutputdir/TUBerlinDataset --tar_dir youroutputdir/TUBerlinDataset.tar
```




## Obtaining the datasets


|     | Paper/Name                                                          | Number of samples | Paper link                                             | Dataset Link                                                                                                       |
| --- | ------------------------------------------------------------------- | ----------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| 1   | Instance Segmentation of Scene Sketches  Using Natural Image Priors | 11234 = 5,617 * 2 | [link](https://inklayer.github.io/)                    | [link](https://www.inkscenes-dataset.com/download)                                                                 |
| 2   | TU Berlin How do humans sketch objects? 2012                        | 20'000            | [link](https://dl.acm.org/doi/10.1145/2185520.2185540) | [link](https://web.archive.org/web/20250131215015/https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) |
| 3   | Sketch-RNN Quickdraw Dataset                                                   | 24’150’000       | -                                                      | [link](https://github.com/googlecreativelab/quickdraw-dataset)                                                     |
| 4   | Ge 2021 Creative Sketch Generation                                  | 167'023           | [link](https://arxiv.org/abs/2011.10039)               | [link](https://songweige.github.io/projects/creative_sketech_generation/gallery_creatures.html)                    |



To obtain the Quickdraw dataset, you will need the gsutil tool. Then you can proceed as follows:
```
mkdir quickdraw
gsutil ls gs://quickdraw_dataset/sketchrnn/*.npz | grep -v '\.full\.npz$' | gsutil -m cp -I quickdraw
```

