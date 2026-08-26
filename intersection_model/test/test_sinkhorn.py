import torch
import sys
import os
top_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_path)
import mymodel

i = float('-inf')
A = torch.tensor([
    [9, 10, 50, 1],
    [10, 2, -25, i],
    [4, 5, 8, 4],
    [1, i, 4,  5]
], dtype=torch.float32)
A.requires_grad = True
A.retain_grad()

res = mymodel.log_sinkhorn_iterations2(A.unsqueeze(0), epsilon=1, num_iters=20).squeeze()
print()
print(res.sum(dim=1), res.sum(dim=0))
print(res)
s = res.sum()
s.backward()
print(A.grad)