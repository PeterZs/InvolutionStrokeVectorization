import torch
import sys
import os
top_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_path)
import mymodel

i = float('-inf')
A = torch.tensor([
    [9, 10, 25, i],
    [10, 2, 24, 3],
    [25, 24, 8, 4],
    [i, 3, 4,  5]
], dtype=torch.float32)
A.requires_grad = True
A.retain_grad()

# res = mymodel.log_sinkhorn_iterations2(A.unsqueeze(0), epsilon=1, num_iters=20).squeeze()
# print()
# print(res.sum(dim=1), res.sum(dim=0))
# print(res.detach())

res2 = mymodel.gumbel_sinkhorn(A.unsqueeze(0), tau=2).squeeze()
print(res2.sum(dim=1), res2.sum(dim=0))

print(res2.detach())