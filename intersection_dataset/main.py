import matplotlib.pyplot as plt
import matplotlib
import sampler.line_sampler as line_sampler
import plotutils
import numpy as np
import cv2
import sampler.intersections as intersections
import rasterizer
import random
import argparse
import os
import sys
from PIL import Image
from pprint import pprint
import csv
import itertools
import utils


matplotlib.rcParams['path.simplify'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="Generate geometric samples.")
    
    parser.add_argument('--out_dir', type=str, default="samples",
                        help='Base output directory.')
    
    parser.add_argument('--npoints', type=int, nargs=2, required=True,
                        help='Min and Max (inclusive) for number of points (discrete).')
    
    parser.add_argument('--closeness', type=float, nargs=2, required=True,
                        help='Min and Max for closeness (continuous).')
    
    parser.add_argument('--target_dim', type=int, default=500,
                        help='Target dimension for the image/canvas.')
    
    parser.add_argument('--from_seed', dest='start_seed', type=int, required=True,
                        help='Starting seed (inclusive).')
    
    parser.add_argument('--to_seed', dest='end_seed', type=int, required=True,
                        help='Ending seed (inclusive).')
    
    parser.add_argument('--n_add', type=int, nargs=2, default= [0, 0],
                        help='Min and Max (inclusive) for number of extra random lines to add')
    
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    # Validation
    # Validate npoints: min 3 points and max 100
    if not (3 <= args.npoints[0] <= args.npoints[1] <= 100):
        print(f"Error: npoints must be >= 3 and <= 100, and min <= max. Got {args.npoints}")
        sys.exit(1)

    # Validate closeness: min 3
    if not (3 <= args.closeness[0] <= args.closeness[1]):
        print(f"Error: closeness must be >= 3, and min <= max. Got {args.closeness}")
        sys.exit(1)
        
    if args.start_seed > args.end_seed:
        print(f"Error: from_seed ({args.start_seed}) cannot be greater than to_inclusive ({args.end_seed})")
        sys.exit(1)

    return args

def ensure_directories(base_dir):
    subdirs = ["vector", "img", "segmentation", "mat", "half_edges_vis", "half_edges_pairs", "lines_json"]
    for sd in subdirs:
        os.makedirs(os.path.join(base_dir, sd), exist_ok=True)



def main():
    args = parse_args()
    
    ensure_directories(args.out_dir)
    
    DIM = args.target_dim
    MIN_DIST = 2
    
    stats_dir = os.path.join(args.out_dir, "stats.csv")
    if os.path.exists(stats_dir): os.unlink(stats_dir)
    

    for seed_idx in range(args.start_seed, args.end_seed + 1):
        attempt = 0
        valid_sample_found = False
        
        print(f"Processing Seed ID: {seed_idx}...")
        
        random.seed(seed_idx)
        np.random.seed(seed_idx)
        while not valid_sample_found:

            # Sample parameters from distributions
            n = random.randint(args.npoints[0], args.npoints[1])
            cl = random.uniform(args.closeness[0], args.closeness[1])
            n_add = random.randint(args.n_add[0], args.n_add[1])
            
            # Attempt generation
            polylines, res, red_points, stats = line_sampler.sample_tri_method(
                DIM, DIM, n_points=n, closeness=cl, min_dist=MIN_DIST, n_add=n_add
            )
            ipoints_per_pl = [np.array([x.point for x in i]) for i in res]
            sequences = rasterizer.rasterized_split_seqs(polylines, ipoints_per_pl, (DIM, DIM))
            # for k in sequences:
            #     pprint(k)
            #     print()
            half_edges = rasterizer.get_half_edge_splits(sequences)
            
            # Get segmentation image
            segmentation_img = np.zeros((DIM, DIM, 3), dtype=np.uint8)
            rasterizer.color_half_edges_for_model(segmentation_img, half_edges)
            
            split_valid = rasterizer.check_numbers(segmentation_img, ipoints_per_pl, polylines)
            half_edges_valid = rasterizer.are_half_edges_valid(half_edges)
            if not split_valid or not half_edges_valid:
                print("invalid sample, retry")
                # Soft fail, retry
                attempt += 1
                continue 
            
            all_ipoints = intersections.points_flattened(ipoints_per_pl)
            n = sum(len(x) for x in sequences)
            half_edge_lines = rasterizer.polyline_half_edges(polylines, res, n)
            h1 = sum(len(x) for x in half_edges)
            h2 = len(half_edge_lines)
            print("num half edges rasterized", h1)
            print("num half edge lines", h2)
            if h1 != h2:
                print("actually invalid sample, retry")
                attempt +=1
                continue
            valid_sample_found = True
            # Format closeness for filename to avoid massive floats
            cl_str = f"{cl:.2f}".replace(".", ",")
            name = f"{seed_idx}_{n}_{cl_str}_{seed_idx}"
            
            # 1. Append sample stats
            utils.append_dict_to_csv(stats_dir, stats, {"name": name})
            
            
            # 4. Adjacency matrix
            M = rasterizer.adjacency_mat(half_edges)
            mat_path = os.path.join(args.out_dir, "mat", f"{name}.csv")
            np.savetxt(mat_path, M, delimiter=",", fmt="%d")

            
            #assert len(half_edge_lines) == M.shape[0]
            
            # 5. half-edge info
            pair_path = os.path.join(args.out_dir, "half_edges_pairs", f"{name}.csv")
            utils.dict_to_kv_csv(rasterizer.half_edge_pairs(half_edges), pair_path)
            
            # Half-edge polylines
            linepath= os.path.join(args.out_dir, "lines_json", f"{name}.json")
            utils.save_polylines_to_json(half_edge_lines, linepath)
            
            # 2. Save Segmentation Image
            img_path = os.path.join(args.out_dir, "segmentation", f"{name}.webp")
            utils.save_webp(segmentation_img, img_path)
            
            # 3. Antialised lines ("drawing")
            img = np.zeros((DIM, DIM, 3), dtype=np.uint8)
            for pl in polylines:
                rasterizer.draw_polyline_cv2(img, pl, (255, 255, 255), aa=True)
            img_path = os.path.join(args.out_dir, "img", f"{name}.webp")
            utils.save_webp(img, img_path)
            
            #print(sorted(list(half_edge_lines.keys())))
                    
            # 5. Plot lines only
            path = os.path.join(args.out_dir, "vector", f"{name}.svg")
            with plotutils.ImageOverlay(path, dims=(DIM, DIM)) as ax:
                plotutils.plot_lines_and_points(ax, lines=[*half_edge_lines.values()],
                                                color=plotutils.COLORS, lw=0.5, 
                                                points= red_points,
                                                linealpha=1, pointcolor="grey")  
                plotutils.plot_lines_and_points(ax, points =all_ipoints, pointcolor="black", psize=1)
            
 

            # 6. Plot half-edges from adjacency matrix
            pos = rasterizer.half_edge_centers(half_edges)
            vis_path = os.path.join(args.out_dir, "half_edges_vis", f"{name}.svg")
            plotutils.plot_half_edges(
                DIM, M, half_edges, polylines, all_ipoints, pos, vis_path, annotate=True)
            
            
            if args.debug:
                #plot splitted lines (complete edges) as well
                path =  os.path.join(args.out_dir, "half_edges_vis", f"{name}_lines.svg")
                plotutils.plot_splitlines(DIM, polylines, sequences, all_ipoints, path, d=MIN_DIST*2)
                
                #test that we can reconstruct the individual paths
                d = utils.kv_csv_to_dict(pair_path)
                polylines = rasterizer.reconstruct_paths(d, M)
                new_img = segmentation_img.copy()
                color_gen = (plotutils.color_to_bgr(x) for x in itertools.cycle(plotutils.COLORS))
                rasterizer.reconstruct_image(polylines, new_img, color_gen)
                
                img_path = os.path.join(args.out_dir, "half_edges_vis", f"{name}_reconstructed.webp")
                utils.save_webp(new_img, img_path)
            
            
            print(f"  -> Success on attempt {attempt}. Saved as {seed_idx}_{n}_{cl_str}_{seed_idx}")
            
            comp = utils.count_components(M)
            print(len([x for x in comp if x > 1]))
            print(len(all_ipoints))
            
            



if __name__ == "__main__":
    main()