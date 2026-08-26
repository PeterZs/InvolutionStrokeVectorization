import metrics_and_losses
import torch

A = torch.tensor([[1, 0, 0, 0],
 [0, 0, 1, 0],
 [0, 1, 0, 0],
 [0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

lengths = torch.tensor([4, 3], dtype=torch.int32)
l= metrics_and_losses.masked_cross_entropy_from_probs(A, A, lengths)
print(l)