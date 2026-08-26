import os
import torch
from pathlib import Path
import cv2
import numpy as np
import sys
import os
import itertools
import argparse
import copy
top_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_path)
import utils
import metrics_and_losses
import dataset
import litmodel
import mymodel
import runconfig
utils.load_env()
utils.set_seed(42)

OUTPATH = "debug_files"



def compare(path_litmodel: str, path_scriptmodel: str):
    dataset_path = Path(os.environ["DATA_ROOT"]) / "train"/"Dataset2"
    d =  {"SampleDataset": str(dataset_path)}
    B = 1
    train_data, _ = dataset.get_dataloaders(d,batch_size=B, val_split=0, num_workers=0)
    config = runconfig.RunConfig()

    model = mymodel.SceneGraphImageModel(max_objects=1000, d_dim=32)
    model1 = litmodel.LitModelWrapper.load_from_checkpoint(path_litmodel, model=model, config = config, map_location="cpu")
    #model1 = litmodel.LitModelWrapper(model=model, config = config).to(torch.device("cpu"))
    model2 = torch.jit.load(path_scriptmodel, map_location="cpu")
    # model2_prog = torch.export.load(path_scriptmodel)
    # model2 = model2_prog.module()    
    model1.eval()
    # model2.eval()
    for idx, batch in enumerate(itertools.islice(train_data, 1)):
        print(idx, batch["basename"])
        n = batch["n_objects"][0]
        seg = batch["seg"][0]
        pairs = batch["pairs"][0]
        target_mat = batch["target_mat"][0]
        compare = [] 
        del batch["basename"]
        for name, currmodel in [("ptmodel", model1), ("scriptmodel", model2)]:
            preds = currmodel(batch)[0]
            print(preds[1][1], preds[2][2])
            compare.append(torch.nan_to_num(preds, nan=0, posinf=0, neginf=0).detach())
            bestM = metrics_and_losses.get_optimal_assignment(preds, n)
            img_vis = metrics_and_losses.reconstruct_image(
                pairs, bestM, seg, n, color_fmt="bgr")
            img_vis_gt = metrics_and_losses.reconstruct_image(
                pairs, target_mat, seg, n, color_fmt="bgr")
            img_pair_vis = metrics_and_losses.construct_pair_img(bestM,  seg,  n, color_fmt="bgr")
            img_pair_vis_gt = metrics_and_losses.construct_pair_img(target_mat,  seg,  n, color_fmt="bgr")
            cv2.imwrite(f"{OUTPATH}/{idx}_{name}pred.png", img_vis)
            cv2.imwrite(f"{OUTPATH}/{idx}_gt.png", img_vis_gt)
            cv2.imwrite(f"{OUTPATH}/{idx}_{name}_predpairs.png", img_pair_vis)
            cv2.imwrite(f"{OUTPATH}/{idx}_gt_pairs.png", img_pair_vis_gt)
            utils.save_matrix(preds, n, f"{OUTPATH}/{idx}_{name}.csv")
        
        diff = compare[0]-compare[1]
        maxdiff = diff.abs().max()
        print("max diff", maxdiff)
        # utils.save_matrix(diff.clone().detach(), n, f"{OUTPATH}/{idx}_diff.csv")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run training or evaluation")

    parser.add_argument(
        "model1",
        type=str,
        help="pytorch lightning model checkpoint",
    )
    parser.add_argument(
        "model2",
        type=str,
        help="traced torchscript model path",
    )
    args = parser.parse_args()
    
    compare(args.model1, args.model2)