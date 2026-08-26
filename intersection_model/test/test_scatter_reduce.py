import torch
import mymodel

torch.manual_seed(42)

H, W = 7, 7
C = 2
K = 5
B = 1

x = torch.arange(float(W)).repeat(C, H, 1).unsqueeze(0)
assert x.shape == (B, C, H, W)
#set second channel to smth else
x[0][1]*=0.5
x.requires_grad = True
print(x)
x.retain_grad()

seg = torch.zeros((B, H, W), dtype=torch.int32, requires_grad=False)
seg[0][0] = 1
seg[0][4] = 2
print(seg)

#[B, K, C]
res = mymodel.scatter_2d(x, seg, K)
print(res.shape)
print(res)

#let's just sum for the sake of debug
s = res.sum()
s.backward()
print(x.grad)


