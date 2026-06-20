# Gradient basics

import torch 
import math 
from einops import einsum, rearrange, reduce, repeat

def oneFwdBkwdPass() :
    # Simple Forward pass

    x = torch.tensor([1., 2,3])
    w = torch.tensor([1., 1,1], requires_grad=True)

    print(x)
    print(w)

    pred_y = x@w # dot product
    loss = 0.5*(pred_y - 5).pow(2)


    # Simple Backward pass

    loss.backward()
    assert loss.grad is None
    assert pred_y.grad is None
    assert x.grad is None
    print(w.grad)
    assert torch.equal(w.grad, torch.tensor([1,2,3]))
    
def cuda_if_available():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numberOfFlopsForGradients():
    B = 1024
    D = 256
    x = torch.randn(B,D, device = cuda_if_available())
    w1 = torch.randn(D,D, device = cuda_if_available(), requires_grad=True)
    w2 = torch.randn(D,D, device = cuda_if_available(), requires_grad=True)
    
    #Forward Pass
    h1 = einsum(x,w1, "batch in, in out -> batch out") # h1 = x@w1
    h2 = einsum(h1,w2, "batch in, in out -> batch out") # h2 = h1@w2
    loss = (h2.mean() - 0)**2
    
    #Backward Pass
    h1.retain_grad()
    h2.retain_grad()
    loss.backward()
    
    #assert gradients
    #need to do
    
    
    
#oneFwdBkwdPass()
numberOfFlopsForGradients()