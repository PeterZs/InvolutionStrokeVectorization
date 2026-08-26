import os
import torch
from pathlib import Path
import torch.nn.functional as F

import dataset
import litmodel
from runconfig import RunConfig
import utils
import permutations
import mymodel
import numpy as np
utils.load_env()
utils.prepare_system()
utils.set_seed(42)
ATOL = 1e-6
np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, linewidth=100)

def test_transformer():
    BATCHSIZE = 2
    DDIM = 4
    SEQ_LEN = 3
    trans = mymodel.RelationSetTransformer(DDIM)
    #IMPORTANT: this disables dropout which is necessary because it is random
    trans.eval()
    perms = permutations.get_batched_padded_perm_matrices(torch.tensor([SEQ_LEN]*BATCHSIZE), 
                                                          perm_func= lambda x: torch.randperm(x),
                                                          pad_size=SEQ_LEN)
    data = torch.rand(BATCHSIZE, SEQ_LEN, DDIM)
    data_perm = torch.bmm(perms, data)
    res = trans(data)
    res_inp_perm = trans(data_perm)
    res_out_perm = torch.bmm(perms, res)
    print(data)
    print(data_perm)
    print(res)
    print(res_inp_perm)
    print(res_out_perm)
    is_close = torch.allclose(res_inp_perm, res_out_perm, atol=ATOL)
    assert is_close

def verify_bins(t1: torch.Tensor, t2: torch.Tensor):
    bins = []
    bins_res = []
    assert t1.max().item() == t2.max().item()
    mx = int(t1.max().item())
    for i in range(mx):
        r1 = [tuple(y.item() for y in x) for x in zip(*torch.where(t1==i))]
        r2 = [tuple(y.item() for y in x) for x in zip(*torch.where(t2==i))]
        bins.append(r1)
        bins_res.append(r2)
    bins.sort()
    bins_res.sort()
    for a, b in zip(bins, bins_res):
        assert a == b

def test_indexmap():
    
    perm = torch.tensor([[0, 1, 0, 0, 0, 0, 0,],
                        [0, 0, 1, 0, 0, 0, 0,],
                        [0, 0, 0, 0, 0, 1, 0,],
                        [1, 0, 0, 0, 0, 0, 0,],
                        [0, 0, 0, 1, 0, 0, 0,],
                        [0, 0, 0, 0, 1, 0, 0,],
                        [0, 0, 0, 0, 0, 0, 1,]], dtype=torch.float32).unsqueeze(0)
    
    MAXINDEX = perm.shape[-1]
    indexes = torch.arange(MAXINDEX).repeat(MAXINDEX, 1).unsqueeze(0)
    msk = torch.triu(torch.ones(MAXINDEX, MAXINDEX)).unsqueeze(0)
    indexes = (indexes * msk).to(torch.int32)
    print(indexes)
    res = permutations.permute_indices(indexes, perm)
    verify_bins(indexes, res)
    
    ifeatures =torch.ones(MAXINDEX, MAXINDEX).unsqueeze(0).unsqueeze(0)
    
    r1 = mymodel.scatter_2d(ifeatures, indexes, MAXINDEX, check_bounds=True)
    r2 = mymodel.scatter_2d(ifeatures, res, MAXINDEX, check_bounds=True)
    r1_perm = torch.bmm(perm, r1)
    print(r1[0].T)
    print(r1_perm[0].T)
    print(r2[0].T)
    assert torch.equal(r1_perm, r2)
    
    


def test_model():
    PAD = 1000
    SMALL = 1000
    HIDDEN_DIM = 4
    BATCH_SIZE = 2
    path = Path(os.environ["DATA_ROOT"]) / "train"/"SampleDataset"
    d =  {"SampleDataset": str(path)}
    train_data, _ = dataset.get_dataloaders(
        d, batch_size=BATCH_SIZE, val_split=0, num_workers=0, max_objects=PAD)
    config = RunConfig(mode="train", model="SceneGraphModel",
                       model_config={"max_objects": SMALL, "d_dim": HIDDEN_DIM})
    model, _ = litmodel.load_lightning_model(config)
    #IMPORTANT: this disables dropout which is necessary because it is random
    model.eval()

    for idx, batch in enumerate(train_data):
        if batch["n_objects"].min() > SMALL:
            continue
        print(idx, batch["basename"], batch["R"].shape)
        batch["R"] = batch["R"][:, :SMALL, :SMALL]
        batch["target_mat"] = batch["target_mat"][:, :SMALL, :SMALL]
        perms = permutations.get_batched_padded_perm_matrices(
            batch["n_objects"], (lambda x: permutations.random_involution(x)), pad_size=SMALL)
        #print("permutation matrix\n", perms.squeeze().cpu().numpy())
        print("bounds: ", batch["seg"].min(), batch["seg"].max())

        
        #verify_bins(batch["seg"], seg_perm)
        
  
        
        res = model(batch)
        out_perm_out = torch.bmm(perms, res)
        #since we output an attention matrix, the columns need to be permuted as well
        out_perm_out = torch.bmm(out_perm_out, perms)
        print()
        print("res permuted from unpermuted input")
        print(out_perm_out.detach().cpu().numpy())
        #again we need to permute rows and colums
        batch["R"] = torch.bmm(torch.bmm(perms, batch["R"]), perms)
        batch["seg"]= permutations.permute_indices(batch["seg"], perms)
        print("bounds: ", batch["seg"].min(), batch["seg"].max())

        out_perm_inp= model(batch)
        
        print("res from permuted input")
        print(out_perm_inp.detach().cpu().numpy())
        assert torch.allclose(out_perm_inp, out_perm_out, atol=ATOL)
        
        break
        


if __name__ == "__main__":
    
    test_transformer()
    test_indexmap()
    test_model()
   