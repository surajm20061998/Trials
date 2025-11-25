import torch

x = torch.randn(3, requires_grad=True)

print(x)

#x.requires_grad_(False)
#y = x.detach()

with torch.no_grad():
    y = x+2
    print(y)
