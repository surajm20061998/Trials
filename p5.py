import torch

weights = torch.randn(4, requires_grad=True)
print(weights)

optimizer = torch.optim.SGD([weights], lr=0.01)
optimizer.step
optimizer.zero_grad()

print (weights)