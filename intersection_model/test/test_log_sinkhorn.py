import torch
import sys
import os
top_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_path)
import mymodel
torch.set_printoptions(precision=8, sci_mode=False)

i = float('-inf')

N = 5  # shape
d = 5.0
pos = torch.randint(0, N-1, (2,))
x = (2 * d) * torch.rand(N, N) - d
x[*pos] = i
s = 0.5*(x + x.T)
s = s.unsqueeze(0)
print(s)

g = mymodel.symmetric_gumbel(s.shape, device=s.device)
print(g)

res = mymodel.log_sinkhorn_iterations2(s, epsilon=10, num_iters=20).squeeze()
print()
print(res.sum(dim=1), res.sum(dim=0))
print(res)