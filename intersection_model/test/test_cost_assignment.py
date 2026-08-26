from mymodel import batched_sinkhorn
from metrics_and_losses import one_hot, involutive_assignment
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx

def hungarian(cost_matrix, maximize = True):
    """
    Apply the Hungarian algorithm to a square cost matrix.

    Args:
        cost_matrix (np.ndarray): shape (N, N), costs to minimize

    Returns:
        row_ind (np.ndarray): row indices of the assignment
        col_ind (np.ndarray): column indices of the assignment
        total_cost (float): sum of assigned costs
    """
    cost_matrix = np.asarray(cost_matrix)
    if maximize:
        cost_matrix = -cost_matrix

    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("cost_matrix must be a square 2D array")

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return row_ind, col_ind, total_cost


A = torch.tensor([
    [7, 10, 8],
    [10, 1, 15],
    [8, 15, 3]
], dtype=torch.float32)
A = torch.tensor([
    [4, 30, 30],
    [30, 1, 15],
    [30, 15, 3]
], dtype=torch.float32)
A = torch.tensor([
    [9, 10, 51, 1],
    [10, 2, 49, 3],
    [51, 49, 8, 4],
    [1, 3, 4,  5]
], dtype=torch.float32)
assert torch.equal(A, A.T)
print(A.numpy())
tmp = one_hot(A)
print(tmp.numpy())
A_stochastic = batched_sinkhorn(A.unsqueeze(0)).squeeze()
print(A_stochastic.numpy())
print(A_stochastic.sum(dim=1), A_stochastic.sum(dim=0))
tmp1 = one_hot(A)
print(tmp1.numpy())
print()
tmp2 = one_hot(A.T).T
print(tmp2.numpy())
print()
row, col, _ = hungarian(A_stochastic, maximize=True)
H = np.zeros_like(A_stochastic, dtype=int)
H[row, col] = 1
print(H)
print()

perm, _ = involutive_assignment(A_stochastic.numpy(), maximize=True)
U = np.zeros_like(A_stochastic, dtype=int)
row = np.arange(len(A_stochastic))
U[row, perm] = 1
print(U)
print()

row, col, _ = hungarian(A, maximize=True)
H = np.zeros_like(A, dtype=int)
H[row, col] = 1
print(H)
print()

perm, _ = involutive_assignment(A.numpy(), maximize=True)
U = np.zeros_like(A, dtype=int)
row = np.arange(len(A))
U[row, perm] = 1
print(U)
print()


Usq = U @ U
print(Usq)
S = A_stochastic @ A_stochastic
print(S)

