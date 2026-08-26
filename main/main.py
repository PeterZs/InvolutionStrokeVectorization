import utils.utils as utils
from PIL import Image, ImageFilter
from graph_extraction.build_linegraph import process, compute_polylines, get_smoothed_lines
# import graph_extraction2
import plotutils.plotutils as plu
import cv2
import centerline
import intersections
import torch
import itertools
import numpy as np
import argparse
from pathlib import Path
import sys
import torch
import imgpreprocess
import traceback
import csv
import matplotlib

matplotlib.rcParams['path.simplify'] = False

def total_length(polylines: list[np.ndarray]) -> float:
    total = 0.0
    for pts in polylines:
        diffs = np.diff(pts, axis=0)
        total += float(np.sum(np.linalg.norm(diffs, axis=1)))
    return total

def apply_pil_blur(pil_img: Image.Image, sigma) -> Image.Image:
    return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image CLI utility"
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input image file or directory containing images",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input)",
    )

    parser.add_argument(
        "--show-more",
        action="store_true",
        help="Save or display intermediate processing results",
    )
    
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay the result on top of the input image",
    )

    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing image",
    )
    parser.add_argument(
        "--noinvert",
        action="store_true",
        help="don't invert (assumes white-on-black image)",
    )
    
    parser.add_argument(
        "--save_lines",
        action="store_true",
        help="save extracted lines as json, don't proceed with intersection resolution",
    )
    
    parser.add_argument(
        "--save_matrix",
        action="store_true",
        help="Save intersection matrix as a csv",
    )
    
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="smooth the lines",
    )
    
    parser.add_argument(
        "--timeit",
        action="store_true",
        help="save runtime to a csv",
    )
    
    parser.add_argument(
        "--sharp",
        action="store_true",
    )
    parser.add_argument(
        "--deg3",
        action="store_true",
    )
    parser.add_argument(
        "--no-deg4",
        action="store_true",
        help="Disable the special degree-4 boundary resolution (enabled by default)",
    )
    parser.add_argument(
        "--ge2",
        action="store_true",
        help="Use the alternative graph_extraction2 pipeline. The "
        "--sharp/--deg3 flags are ignored in this mode.",
    )


    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Input does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
    return args


def process_image(
    image_path: Path,
    output_dir: Path,
    centerline_model: torch.ScriptModule,
    imodel: torch.ScriptModule,
    preprocess: bool = True,
    show_intermediate: bool = False,
    overlay: bool = False,
    invert : bool = True,
    save_lines_only: bool = False,
    save_matrix: bool = False,
    smooth_lines: bool = False,
    sharp = False,
    deg3 = False,
    deg4 = True,
    use_ge2 = False,
):
    path_prefix =  str(output_dir / f"{image_path.stem}_")
    
    image =  Image.open(image_path).convert("L")
    image = imgpreprocess.preprocess(image, invert=invert, skip_norm=not preprocess)
    if show_intermediate:
        image.save(path_prefix + "centerline_input.png")
    print("predict centerline")
    with utils.timer("predict centerlin"):
        out = centerline.predict_centerline(centerline_model, image, invert=False)
    if show_intermediate:
        out.save(path_prefix + "centerline.png")
    cl_img = utils.pil_to_cv2(out)


    print("build graph")
    with utils.timer("GRAPH"):
        # if use_ge2:
        #     g = graph_extraction2.process(
        #         cl_img,
        #         debug_dir=str(output_dir) if show_intermediate else "",
        #         stem=image_path.stem,
        #     )
        # else:
        g, success = process(cl_img, invert=False,
                            debug_dir=str(output_dir) if show_intermediate else "",
                            sharp=sharp,
                            deg3=deg3,
                            deg4=deg4)
    lines, connections= compute_polylines(g)
    if smooth_lines:
        lines = get_smoothed_lines(lines, d = 0.5)
    
    print("total line length:", total_length(lines))
    
    if show_intermediate:
        print("writing polyine svg")
        out_path = path_prefix + "sublines.svg"
        plu.plot_lines_on_img(cl_img, lines, out_path, linewidth=0.4)
    print("\n\n")
    
    if save_lines_only:
        utils.save_polylines_to_json({i: v for i, v in enumerate(lines)}, path_prefix + "lines.json")
        return
    
    for i in range(1):
        
        lines_curr = lines
        #lines_curr = [intersections.add_profile_noise(pl, max_noise_std=0.3, sigma=1) for pl in lines]
        
    
        img_dr, seg_img, R, half_edges, outvis = intersections.build_intersection_segmentation(
            lines_curr, connections, source_size=cl_img.shape[0], target_size=cl_img.shape[0], vis=True)
        
        #img_dr = intersections.add_gaussian_noise(img_dr, std_dev=30)

        #img_dr = utils.pil_to_cv2(apply_pil_blur(Image.fromarray(img_dr), 1))
        if show_intermediate and i ==0:
            cv2.imwrite(path_prefix +  "half_edges.png", outvis)
            cv2.imwrite(path_prefix + "draw.png", img_dr)
            intersections.plot_half_edge_connections(path_prefix+  "connections.svg", half_edges, outvis, R)
        # intersections.verify_seg_img(seg_img, half_edges)


        print("predicting intersection order...")
        #device = 
        with utils.timer("predict"):
            raw_mat= intersections.predict(imodel, img_dr, seg_img, R, half_edges)
        with utils.timer("matching lines"):
            reslines = intersections.get_matched_lines(raw_mat, half_edges)
        if save_matrix:
            #the first row / column is for the background index, so we can just overwrite it
            out = raw_mat.copy()
            tmp = np.arange(0, len(raw_mat))
            out[0] = tmp
            out[:, 0] = tmp
            np.savetxt(path_prefix + "mat.csv", out, delimiter=",", fmt="%.3f")

        out_path = path_prefix + f"result_linenoise{i}.svg"
        back = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if overlay else None
        print("writing result...")
        with plu.ImageOverlay(out_path, back, dims=cl_img.shape) as ax:
            
            # for component_lines in reslines:
            #     c = itertools.cycle(itertools.chain.from_iterable(plu.PAIRED_COLORS))
            #     plu.plot_polylines(ax, component_lines, c)
            
            concat_lines = [np.concat(x) for x in reslines]
            # plu.plot_polylines_rainbow(ax, concat_lines, cmapname="hsv")
            c = itertools.cycle(plu.COLORS)
            plu.plot_polylines(ax, concat_lines, c)
            # for c in concat_lines:
            #     for i in range(0, len(c), 40):
            #         circle = plt.Circle(c[i], 1.5, color="black", transform=ax.transData)
            #         ax.add_patch(circle)
            # for component_lines in reslines:


if __name__ == "__main__":
    args = parse_args()
    utils.load_env()
    np.random.seed(42)
    

    print("loading models...")
    cmodel = centerline.load_model()
    imodel = None
    if not args.save_lines:
        imodel = intersections.load_model()
    output_dir = utils.resolve_output_dir(args.input, args.output_dir)

    images = utils.collect_images(args.input)
    if not images:
        print("No images found to process.", file=sys.stderr)
        sys.exit(1)

    bad_dir = output_dir / "bad"
    runtime_csv = output_dir / "runtime.csv"
    for idx, img_path in enumerate(images):
        print(f"\n\n\n{idx+1}/{len(images)}", "processing", img_path)
        try:
            with utils.timer() as t:
                process_image(
                    img_path,
                    output_dir,
                    cmodel,
                    imodel,
                    preprocess=not args.skip_preprocess,
                    show_intermediate=args.show_more,
                    overlay=args.overlay,
                    invert=not args.noinvert,
                    save_lines_only=args.save_lines,
                    save_matrix=args.save_matrix,
                    smooth_lines=args.smooth,
                    sharp = args.sharp,
                    deg3=args.deg3,
                    deg4=not args.no_deg4,
                    use_ge2=args.ge2,
                )
            if args.timeit:
                with open(runtime_csv, "a", newline="") as f:
                    csv.writer(f).writerow([img_path.name, f"{t['time']:.4f}"])

        except Exception:
            bad_dir.mkdir(parents=True, exist_ok=True)
            (bad_dir / f"{img_path.stem}.txt").write_text(traceback.format_exc())
            print(f"FAILED: {img_path.name} (see {bad_dir / img_path.stem}.txt)", file=sys.stderr)








