import torch
import cv2
import numpy as np
from typing import Dict, Literal
import torch.nn as nn

import torch.distributed as dist
import torch.nn.functional as F
from torchmetrics import Metric

import utils
import networkx as nx

def masked_cross_entropy_from_probs(probs, targets, lengths):
    """
    probs:   [B, K, K] - Already softmaxed (values 0.0 to 1.0).
    targets: [B, K, K] - Ground truth (0.0 or 1.0).
    lengths: [B]       - Int64 tensor of actual object counts per batch item.
    """
    B, K, _ = probs.shape
    
    # 1. Create Mask for Valid Rows (Objects)
    # Shape: [B, K]
    # Creates indices [0, 1, ... K-1] and compares to lengths
    range_vector = torch.arange(K, device=probs.device).unsqueeze(0) # [1, K]
    valid_obj_mask = range_vector < lengths.unsqueeze(1) # [B, K]
    
    # 2. Compute Log Probabilities
    # We add 1e-9 to prevent log(0) which produces NaNs.
    # We also clamp max value to 1.0 just to be safe.
    safe_probs = torch.clamp(probs, min=1e-9, max=1.0)
    log_probs = torch.log(safe_probs)
    
    # 3. Compute Cross Entropy: -Target * log(Prediction)
    # Shape: [B, K, K]
    ce_loss = -targets * log_probs
    
    # 4. Sum over the classes/columns (loss per object)
    # Shape: [B, K]
    loss_per_object = ce_loss.sum(dim=-1)
    
    # 5. Mask out padding objects
    # If valid_obj_mask is False (0), the loss becomes 0.
    active_loss = loss_per_object * valid_obj_mask.float()
    
    # 6. Normalize by Total Number of Objects
    num_valid_objects = valid_obj_mask.sum()
    
    # Avoid division by zero
    return active_loss.sum() / (num_valid_objects + 1e-9)


def batched_identity(sizes, N):
    """
    sizes: (B,) tensor with matrix sizes
    N: int (max matrix size)

    Returns:
        (B, N, N) tensor where each batch element contains
        an identity matrix of size sizes[b] padded with zeros.
    """
    B = sizes.shape[0]

    # Base identity matrix (N, N)
    eye = torch.eye(N, device=sizes.device)

    # Expand to (B, N, N)
    eye_batch = eye.expand(B, N, N).clone()

    # Create mask for valid indices
    idx = torch.arange(N, device=sizes.device)
    valid = idx.unsqueeze(0) < sizes.unsqueeze(1)  # (B, N)

    # Apply mask to rows and columns
    mask = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, N, N)

    return eye_batch * mask


def orthogonal_loss(probs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor):
    A = probs @ probs.transpose(1, 2)
    I = batched_identity(lengths, probs.shape[1])
    return F.mse_loss(A, I)
    

def l2loss(probs, targets, lengths):
    return F.mse_loss(probs, targets)

LOSS_MAP = {
        "MaskedCrossEntropyLoss": masked_cross_entropy_from_probs,
        "MSELoss": l2loss,
        "OrthogonalLoss": orthogonal_loss
    }

def build_loss_func(losses: dict, device: torch.device):
    """
    Build a weighted sum loss function from a dictionary of loss names -> weights.

    Args:
        losses (dict): {"LossName": weight}, supported LossName: "L1", "MSE", "BCE", etc.

    Returns:
        A function loss_fn(preds, targets) -> scalar loss
    """
    # Filter valid losses
    active_losses = {}
    for name, weight in losses.items():
        if name in LOSS_MAP and weight > 0:
            active_losses[name] = (LOSS_MAP[name], weight)
        else:
            raise ValueError(f"Unsupported loss '{name}' or unvalid weight: '{weight}")

    # Return a function that computes weighted sum
    def loss_fn(*args):
        total_loss = torch.tensor(0.0, device=device)
        components = {}
        for name, (loss_func, weight) in active_losses.items():
            curr = loss_func(*args)
            components[name] = curr
            total_loss += weight * curr
        return total_loss, components

    return loss_fn




def one_hot(x: torch.Tensor):
    # x: (B, N, M)
    indices = x.argmax(dim=-1)               # (B, N)
    return F.one_hot(indices, num_classes=x.size(-1))



def involutive_assignment(C: np.ndarray, maximize: bool = True):
    """
    Find a maximum- or minimum-cost involutive assignment.

    This function constructs a graph of size 2*N to transform the problem
    into a Maximum Weight Perfect Matching problem. 
    
    Nodes 0 to n-1 represent the 'Real' layer.
    Nodes n to 2n-1 represent the 'Shadow' layer.
    
    - Fixed points (C[i,i]) are modeled as edges between Real i and Shadow i.
    - Swaps (C[i,j]) are modeled as edges (i, j) in Real and (i+n, j+n) in Shadow.
    
    The solver finds a perfect matching that maximizes the sum of weights.
    Due to the symmetric construction, the optimal matching in the Real layer 
    mirrors the Shadow layer, providing the correct total cost for swaps 
    (C[i,j] + C[j,i]) without manually summing them beforehand.

    Parameters
    ----------
    C : np.ndarray (n x n)
        Symmetric cost matrix. C[i,j] == 0 means forbidden.
    maximize : bool
        If True, maximize total cost. If False, minimize.

    Returns
    -------
    perm : np.ndarray
        Involutive permutation array of length n.
    total_cost : float
        Total assignment cost.
    """
    n = C.shape[0]
    G = nx.Graph()
    

    def get_weight(val):
        return val if maximize else -val

    # Construct the 2N graph
    for i in range(n):
        # 1. Edge for Fixed Point: Connect Real i to Shadow i (i <-> i+n)
        # Weight is C[i, i]
        w_fixed = get_weight(C[i, i])
        G.add_edge(i, i + n, weight=w_fixed)
        
        # 2. Edges for Swaps: Connect i <-> j within layers
        for j in range(i + 1, n):
            if C[i, j] == 0.0 or not np.isfinite(C[i, j]):
                continue
            
            w_swap = get_weight(C[i, j])
            
            # Add edge in Real layer
            G.add_edge(i, j, weight=w_swap)
            
            # Add edge in Shadow layer
            G.add_edge(i + n, j + n, weight=w_swap)

    # Solve Maximum Weight Perfect Matching
    # maxcardinality=True ensures the matching covers as many nodes as possible 
    # (conceptually a Perfect Matching for this construction).
    matching = nx.max_weight_matching(G, maxcardinality=True)

    # Reconstruct the permutation
    perm = np.arange(n, dtype=int)
    
    # We only need to parse the matching relative to the Real layer (0 to n-1)
    for u, v in matching:
        # Normalize so u is smaller
        if u > v:
            u, v = v, u
            
        # Case 1: Fixed Point Edge (u in Real, v in Shadow)
        # u < n and v >= n. Specifically v should be u + n.
        if v == u + n:
                perm[u] = u
        
        # Case 2: Swap Edge in Real Layer (u < n and v < n)
        # Note: We ignore Swap Edges in Shadow Layer (u >= n and v >= n)
        elif u < n and v < n:
            perm[u] = v
            perm[v] = u

    # Calculate exact total cost based on the resulting permutation
    total_cost = 0.0
    for i in range(n):
        total_cost += C[i, perm[i]]

    return perm, total_cost


def get_optimal_assignment(raw_mat: torch.Tensor, n):
    mat = raw_mat.detach().cpu()[:n, :n].numpy()
    perm, _ = involutive_assignment(mat)
    R = torch.zeros(raw_mat.shape, dtype=torch.int32)
    row = np.arange(len(mat))
    R[torch.tensor(row), torch.tensor(perm)] = 1
    return R


def upper_triangle_loss1(predM: torch.Tensor, targetM: torch.Tensor, n: torch.Tensor, constraintM: torch.Tensor):
    """
    predM, targetM: (batch_size, N, N)
    n: (batch_size,) tensor of ints, number of rows/cols to consider per sample
    """
    batch_size, N, _ = predM.shape
    
    # Create base upper triangular mask (N x N)
    base_mask = torch.triu(torch.ones(N, N, device=predM.device))
    
    # Create row/col index tensors (N,)
    row_idx = torch.arange(N, device=predM.device).view(1, N, 1)
    col_idx = torch.arange(N, device=predM.device).view(1, 1, N)
    
    # Broadcast n for comparison
    n_expanded = n.view(batch_size, 1, 1)
    
    # Mask: upper triangle AND within first n rows and n cols
    mask = (row_idx < n_expanded) & (col_idx < n_expanded) & (base_mask.bool())
    
    # Element-wise multiply and apply mask
    mult = predM * targetM * mask.float()
    
    # Sum over last two dimensions
    sum_mult = mult.sum(dim=(1, 2))
    
    # Divide by n per sample
    loss = sum_mult / n.float()
    
    return loss



def upper_triangle_loss(predM: torch.Tensor, targetM: torch.Tensor, n: torch.Tensor, constraintM: torch.Tensor):
    """
    predM, targetM: (batch_size, N, N)
    n: (batch_size,) tensor of ints, number of rows/cols to consider per sample
    constraintM: (batch_size, N, N) or (batch_size, N, K)
                 Used to filter rows. If a row has exactly 1 element > 0, 
                 it is disregarded from the loss.
    """
    batch_size, N, _ = predM.shape
    device = predM.device
    assert n.ndim == 1
    
    # --- 1. Identify rows to disregard ---
    # Count non-zero elements per row in constraintM
    # Shape: (batch_size, N)
    constraint_counts = (constraintM > 0).float().sum(dim=-1)
    
    # Boolean mask where rows have exactly 1 constraint (True = disregard this row)
    # Shape: (batch_size, N)
    bad_rows = (constraint_counts == 1)
    
    # --- 2. Create Masks ---
    
    # Base structural masks
    base_mask = torch.triu(torch.ones(N, N, device=device))
    row_idx = torch.arange(N, device=device).view(1, N, 1)
    col_idx = torch.arange(N, device=device).view(1, 1, N)
    n_expanded = n.view(batch_size, 1, 1)
    
    # Filter A: Upper triangle AND within first n rows/cols
    # Shape: (batch_size, N, N)
    structure_mask = (row_idx < n_expanded) & (col_idx < n_expanded) & (base_mask.bool())
    
    # Filter B: Filter out bad rows
    # We broadcast bad_rows from (B, N) to (B, N, 1) to mask the entire row 'i' across all columns
    valid_row_mask = (~bad_rows).view(batch_size, N, 1)
    
    # Combine masks
    final_mask = structure_mask & valid_row_mask
    
    # --- 3. Compute Numerator (Sum) ---
    # Element-wise multiply and apply mask
    # Rows flagged as bad are now 0 in the mask, so they contribute 0 to the sum
    mult = predM * targetM * final_mask.float()
    
    # Sum over dimensions 1 and 2
    sum_mult = mult.sum(dim=(1, 2))
    
    # --- 4. Compute Denominator (n) ---
    # We must subtract the count of bad rows, but ONLY if those rows were inside the 'n' range.
    
    # Mask indicating which indices are effectively used by 'n'
    # Shape: (batch_size, N)
    idx_range_mask = torch.arange(N, device=device).view(1, N) < n.view(batch_size, 1)
    
    # Identify bad rows that fall within the valid n range
    effective_bad_rows = bad_rows & idx_range_mask
    
    # Count how many rows to subtract per sample
    subtract_counts = effective_bad_rows.float().sum(dim=1)
    
    # Adjust n
    n_adjusted = n.float() - subtract_counts
    
    # Avoid division by zero (optional safety)
    n_adjusted = torch.clamp(n_adjusted, min=1.0)
    
    return sum_mult, n_adjusted

def unique_row(constraintM: torch.Tensor):
    constraint_counts = (constraintM > 0).float().sum(dim=-1)
    # Boolean mask where rows have exactly 1 constraint (True = disregard this row)
    # Shape: (batch_size, N)
    bad_rows = (constraint_counts == 1)
    return bad_rows.sum(dim=-1)

def count_symmetric_assignments(target: torch.Tensor):
    unique_matches = torch.triu(target, diagonal=0)
    return unique_matches.sum(dim=(1, 2))

def count_correct_symmetric_assignments(
    A: torch.Tensor, 
    B: torch.Tensor, 
) -> torch.Tensor:
    """
    Computes the number of correct assignments for a batch of symmetric permutation matrices.
    Assumes that any values in A or B outside the valid range n are strictly 0.
    
    Args:
        A (torch.Tensor): Predicted matrices. Shape (Batch, N, N).
        B (torch.Tensor): Target matrices. Shape (Batch, N, N).
        n (torch.Tensor): (Optional) Number of rows/cols. Not needed for calculation 
                          due to zero-padding assumption, but kept for API compatibility.
        
    Returns:
        torch.Tensor: A tensor of shape (Batch,) containing the count for each sample.
    """

    matches = A * B
    
    # 2. Handle Symmetry (Avoid Double Counting)
    # triu keeps the diagonal (self-loops) and the upper triangle (unique edges).
    # It sets the lower triangle to 0.
    unique_matches = torch.triu(matches, diagonal=0)
    
    # 3. Sum
    # Sum over height (dim 1) and width (dim 2) to get total counts per batch.
    return unique_matches.sum(dim=(1, 2))



def indexed_color(i: int) -> tuple[int, int, int]:
    """
    Encode an integer index into an RGB color using base-256.

    Parameters
    ----------
    i : int
        Layer index (0 <= i < 256**3)

    Returns
    -------
    (r, g, b) : tuple[int, int, int]
        RGB color encoding the index
    """
    if i < 0 or i >= 16_777_216:
        raise ValueError("Index out of range (must be 0 <= i < 256^3)")

    r = i % 256
    g = (i // 256) % 256
    b = (i // (256**2)) % 256

    return r, g, b


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        # If they are already in the same component, do nothing
        if root_i == root_j:
            return False

        # Union by Rank: attach smaller tree to larger tree
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            # If ranks are same, attach one to other and increment rank
            self.parent[root_j] = root_i
            self.rank[root_i] += 1
            
        return True

def reconstruct_paths(half_edge_pairs: np.ndarray, M: np.ndarray, debug = False):
    #print("input dim", M.shape, len(half_edge_pairs))
    uf = UnionFind(M.shape[0])
    #skip 0
    for idx in range(1, M.shape[0]):
        #correct because we know there is only 1 in the row
        col_idx = np.flatnonzero(M[idx]==1)[0]
        uf.union(idx, col_idx)
        uf.union(idx, half_edge_pairs[idx])
        if debug:
            print("union", idx, col_idx, half_edge_pairs[idx])
    
    components = {}
    [components.setdefault(uf.find(k), []).append(k) for k in range(len(M))]
    return components

def reconstruct_ordered_paths(nodes, m1, m2) -> list[list[int]]:
    visited = set()
    paths = []

    def next_node(curr, prev):
        """Return the next node from curr that is not prev (if possible)."""
        a, b = m1[curr], m2[curr]
        # Choose endpoint that is not prev
        if a != prev and a != curr:
            return a
        if b != prev and b != curr:
            return b
        return None

    for start in nodes:
        if start in visited:
            continue

        # Build path starting from 'start'
        path = [start]
        visited.add(start)

        prev = None
        curr = start
        for _ in range(0, len(nodes)):
            nxt = next_node(curr, prev)
            if nxt is None or nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        prev = None
        curr = start
        for _ in range(0, len(nodes)):
            nxt = next_node(curr, prev)
            if nxt is None or nxt in visited:
                break
            path.insert(0, nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        paths.append(path)

    return paths




def reconstruct_image_from_components(components: list[list[int]], seg_img: np.ndarray, color_fmt = "rgb"):
    assert seg_img.ndim ==2
    img = np.ones((seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)*255
    color_iter = utils.color_gen(rgb=color_fmt=="rgb")
    for v in components:
        #skip background
        if 0 in v:
            assert len(v) ==1
            continue
        new_color = next(color_iter)
        for idx in v:
            mask = seg_img == idx
            img[mask] = np.array(new_color)
    return img

def reconstruct_image(pairs: torch.Tensor, assignment: torch.Tensor, raw_seg_img: torch.Tensor, n, debug=False, color_fmt = "rgb"):
    #note that n already includes the background
    n = int(n)
    assert len(assignment) >= n
    assert assignment.ndim == 2
    assert raw_seg_img.min() == 0
    mat = assignment[:n, :n].cpu().numpy()
    perm = {idx: int(np.flatnonzero(mat[idx]==1)[0]) for idx in range(1, n)}
    nodes = np.arange(1, n)
    components =reconstruct_ordered_paths(nodes, pairs.cpu().numpy(), perm)
    if debug:
        print(components)
    return reconstruct_image_from_components(components, raw_seg_img.cpu().numpy(), color_fmt=color_fmt)


def construct_pair_img(raw_mat: torch.Tensor, raw_seg_img: torch.Tensor, n, color_fmt = "rgb"):
    assert raw_mat.ndim ==2
    assert raw_seg_img.min() == 0
    assert raw_seg_img.max()+1 == n
    seg = raw_seg_img.cpu().numpy()
    mat = raw_mat.cpu().numpy()[:n, :n]
    #print(mat)
    #assert np.array_equal(mat, mat.T)
    img = np.ones((raw_seg_img.shape[0], raw_seg_img.shape[1], 3), dtype=np.uint8)*255
    color = utils.paired_color_gen(rgb= color_fmt =="rgb")
    
    # skip 0
    for idx in range(1, n):
        col_idx = np.flatnonzero(mat[idx]==1)[0]
        if col_idx == idx:
            mask = seg == idx
            img[mask] = utils.color_to_bgr("grey")
        if col_idx > idx:
            a, b = next(color)
            img[seg == idx] = a
            img[seg == col_idx] = b
    return img
            


def ce_metric(batch: dict, preds: torch.Tensor):
    return masked_cross_entropy_from_probs(preds, batch["target_mat"], batch["n_objects"])

def correct_connections(batch: dict, preds: torch.Tensor):
    res = []
    for mat, n_i in zip(preds, batch["n_objects"]):
        res.append(get_optimal_assignment(mat, n_i))
    result = torch.stack(res).to(preds.device)
    assert result.shape == preds.shape
    u = unique_row(batch["R"])
    n = count_correct_symmetric_assignments(result, batch["target_mat"])
    init = count_symmetric_assignments(batch["target_mat"])
    r = (n-u) / torch.clamp(init-u, min=1.0)
    r[init == u] = 1.0
    return r



class GlobalStatsMetric(Metric):
    full_state_update = False

    def __init__(self, metric_func, smaller_is_better: bool, **kwargs):
        super().__init__(**kwargs)
        print("init called")
        # 1. Store ALL scalar scores to compute Mean/Median/Std/etc.
        # dist_reduce_fx="cat" will concatenate results from all ranks automatically.
        self.add_state("scores", default=[], dist_reduce_fx="cat")

        # 2. Local tracking for the "Worst" sample (tensors).
        # We handle this manually in Python to avoid complex TorchMetrics logic.
        self._reset_local_tracking()
        self.metric_func= metric_func
        self.compare_best = lambda a, b : a <= b if smaller_is_better else a >= b
        self.smaller_is_better = smaller_is_better
        

    def _reset_local_tracking(self):
        #self.worst_score = float("-inf") if self.smaller_is_better else float("inf")
        self.worst_score = None
        # This will hold a dictionary of tensors for the single worst sample, e.g.:
        # {'input': Tensor, 'target': Tensor, 'pred': Tensor} (all on CPU)
        self.worst_sample_data = None

    def reset(self):
        super().reset()
        self._reset_local_tracking()

    def update(self, metric_args_inp: tuple, metric_args_pred: tuple, sample_data: Dict[str, torch.Tensor | str]):
        """
        metric_values: Shape (Batch,) - The score for each sample (higher is "worse" or "better" depending on your logic)
        sample_data: Dict of batches, e.g. {'img': (B,C,H,W), 'pred': (B, ...)}
        """
        # --- 1. Accumulate Scores for Global Stats ---
        # Detach and move to CPU immediately
        current_scores = self.metric_func(metric_args_inp, metric_args_pred).detach().flatten().cpu()
        #print(current_scores)
        self.scores.append(current_scores)

        # --- 2. Update Local "Worst" Sample ---
        # Find the max score in this specific batch
        batch_worst_val, batch_idx = torch.max(current_scores, dim=0) if self.smaller_is_better else torch.min(current_scores, dim=0) 
        batch_worst_val = batch_worst_val.item()
        if self.worst_score is None:
            self.worst_score = batch_worst_val

        # If this batch contains a new global max (for this rank), overwrite our stored sample
        if self.compare_best(self.worst_score, batch_worst_val):
            self.worst_score = batch_worst_val

            # Extract the specific slice corresponding to batch_idx and move to CPU
            # We construct a new dict containing only that single sample
            self.worst_sample_data = {}
            for k, v in sample_data.items():
                if not isinstance(v[batch_idx], torch.Tensor):
                    self.worst_sample_data[k] = v[batch_idx]
                else:
                # v[batch_idx] extracts the specific sample
                    self.worst_sample_data[k] = v[batch_idx].detach().cpu()

    def compute(self):
        # --- 1. Compute Numeric Stats ---
        if not self.scores:
            return None, None

        all_scores = torch.cat(self.scores, dim=0).float()

        results = {
            f"mean": torch.mean(all_scores),
            f"median": torch.median(all_scores),
            f"std": torch.std(all_scores),
            f"min": torch.min(all_scores),
            f"max": torch.max(all_scores),
        }

        # --- 2. Find Global Worst Sample (DDP Sync) ---

        # Current rank's candidate
        global_worst_score = self.worst_score
        global_worst_data = self.worst_sample_data

        if dist.is_available() and dist.is_initialized():
            # Gather candidates from all ranks
            # We package score + data into a simple object
            my_candidate = {
                "score": self.worst_score,
                "data": self.worst_sample_data
            }

            world_size = dist.get_world_size()
            candidates = [None] * world_size

            # all_gather_object handles pickling dicts/tensors automatically
            dist.all_gather_object(candidates, my_candidate)

            # Find the winner among the gathered candidates
            # Reset to find the max in the list
            global_worst_score = float("-inf")

            for cand in candidates:
                #if global_worst_score is better than cand["score"]
                if self.compare_best(global_worst_score, cand["score"]):
                    global_worst_score = cand["score"]
                    global_worst_data = cand["data"]


        return results, global_worst_data