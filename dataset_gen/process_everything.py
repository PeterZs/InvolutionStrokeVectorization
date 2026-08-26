import drawutils
import json
import os
import uuid
import argparse
from dataset_loaders import loader_utils
from typing import Literal, Union
import random
import utils
from itertools import islice
from datetime import datetime

BRUSHES_TO_USE = ["calligraphy_marker_small", "pencil_2", "ink", "pencil 4 ", "pencil_3", "small_pen", "watercolor_2", "watery_ink", "tiny_pen"]
BRUSHES_TO_USE = ["pencil 4", "pencil_3", "pencil_2", "ink_2", "tiny_pen", "small_pen 2", "watery_ink", "calligraphy_marker_small", "marker_1"]


def get_date():

    now = datetime.now()
    # Format it as +%Y-%m-%d_%H-%M-%S
    formatted_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_name


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Generation Script")
    parser.add_argument('--mode', type=str, default='all', help='Mode of operation')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--target_size', type=int, default=1000, help='Target size for the dataset')
    parser.add_argument('--out_dir', type=str, default='', help='Output directory')
    parser.add_argument('--class', dest='class_name', nargs='*', default=[], help='Dataset class names to use')
    parser.add_argument("--offset", type=int, default=0, help='''
                        offset (0-indexed) of the first dataset class sample to generate. But if the samples with indexes offset, ..., c are already generated, the first sample will be c.''')
    parser.add_argument("--to", type=int, default=None, help='''
                        take until index --to samples (inclusive) from the dataset. If not specified, all samples of the dataset will be generated''')
    parser.add_argument("--num_brushes_per_sample", type=int, default= -1, help='''
                        only use that many brushes per sample (chosen randomly among available brushes)''')
    parser.add_argument("--skip_variations", action="store_true", default=False)
    args = parser.parse_args()
    if args.mode not in ['all', 'demo']:
        raise ValueError(f"mode should be one of ['all', 'demo']. default is 'all'")

    if args.to < args.offset:
        raise ValueError(f"--to is smaller than --offset")
    return args



#uv run process_everything.py --num_brushes_per_sample 5 --out_dir dataset_sample/output --k 1
#uv run process_everything.py --out_dir dataset_sample/output --k 2 --class TUBerlinDataset
#uv run process_everything.py --out_dir 'datasets/OUT_TUBERLIN/1_1000' --k 1000 --class TUBerlinDataset
#uv run process_everything.py --out_dir 'datasets/OUT_TUBERLIN/1001_2000' --offset 1000 --k 1000 --class TUBerlinDataset
#uv run process_everything.py --out_dir 'datasets/OUT_TUBERLIN/2000_2999' --offset 2000 --k 1000 --class TUBerlinDataset
#uv run process_everything.py --out_dir 'dataset/OUT_PIMPMYDRAWINGS' --class PimpMyDrawingDataset

#to run a small selection of all samples:
# uv run process_everything.py --out_dir dataset_sample/output --num_brushes_per_sample 5 --offset 100 --to 101

#debug 
# uv run process_everything.py --out_dir dataset_sample/output --offset 1009 --to 1009 --class TUBerlinDataset


if __name__ == '__main__':
    args = parse_args()

    loader_classes = loader_utils.find_loader_classes()
    available_classes = {c.__name__ for c in loader_classes}

    print(f"Running in mode: {args.mode}")
    print(f"Using split: {args.split}")
    print(f"Target size: {args.target_size}")
    print(f"Output directory: {args.out_dir}")
    print(f"Classes chosen: {args.class_name}")
    
    random.seed(42)
    # exit()
    brushes = drawutils.get_all_brushes(include=BRUSHES_TO_USE)
    
    #define which brushes need down/upscaling
    needs_downscaling = ['ink', 'watery_ink', 'pencil_3', "small_pen 2",
                         'marker_1_small', 'marker_1', 'charcoal', 'calligraphy_marker_small', 'watercolor', 'watercolor_2']
    variations = {}
    if not args.skip_variations:
        with open('brush_variations.json', 'r') as f:
            variations = json.load(f)
    appendfile = args.out_dir + "/filelist.jsonl"
    
 
    loader_classes = [x for x in loader_classes if x.__name__ in args.class_name or not len(args.class_name)]
    for L in loader_classes:
        d = L()
        print(d.get_class_name())
        # we don't want to generate pairs for samples already used
        c = utils.get_generated_count_for_class(args.out_dir, d.get_class_name())
        print(c)
        # c is the *current* max sample. 
        # but since we may not have generated all brush outputs for 
        # the current sample, we don't want to do c+1. 
        # So, we end up regenerating outputs for sample c, which get overwritten.
        c = max(c, args.offset)
        upper = args.to-c+1
        if upper < 0:
            continue
        print("starting processing at index", c)
        for svg_sample in islice(d.sample_loader(args.split, args.target_size, index=c), upper):
            print("[" + get_date()+ "]", "processing sample", c)
            drawutils.generate_image_pairs(
                svg_sample, 
                brushes,
                variations, 
                out_dir=args.out_dir, 
                naming = f"{d.get_class_name()}_{c}",
                out_image_size=args.target_size,
                num_variations=3,
                append_log_file=appendfile,
                needs_downscaling = needs_downscaling,
                num_brushes_per_sample=args.num_brushes_per_sample)
            c+=1
        
        
        