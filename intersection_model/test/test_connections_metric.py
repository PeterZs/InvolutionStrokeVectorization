import os
import torch
import utils
from pathlib import Path
import dataset
import metrics_and_losses
utils.load_env()
path = Path(os.environ["DATA_ROOT"]) / "train"/"SampleDataset"
d =  {"SampleDataset": str(path)}
train_data, _ = dataset.get_dataloaders(d,batch_size=2, val_split=0, num_workers=0)


for idx, batch in enumerate(train_data):
    print(idx, batch["basename"])
    
    for mat, n_i in zip( batch["target_mat"], batch["n_objects"]):
        M = metrics_and_losses.get_optimal_assignment(mat, n_i)
        # the optimal assignment
        # of cost matrix = permutation matrix itself must be the permutation matrix
        assert torch.equal(M, mat)
    

    gt = metrics_and_losses.count_symmetric_assignments(batch["target_mat"])
    u = metrics_and_losses.unique_row(batch["R"])
    l = metrics_and_losses.count_correct_symmetric_assignments(
        batch["target_mat"], batch["target_mat"])
    print( l, gt, u)
    r = metrics_and_losses.correct_connections(batch,  batch["target_mat"])
    print(r)
    assert (r == 1.0).all()