#Parameter Calculations

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math

def cuda_if_available():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def deep_network():
    D = 8
    L = 3
    device  = cuda_if_available()
    model = DeepNetwork(dim=D,num_layers=L).to(device)
    num_parameters = get_num_parameters(model)
    assert num_parameters == (D*D)*L 
    B = 4 # Batch Size
    x = torch.randn(B,D,device=device)
    y = model(x)
    
    print("Device : ", device)
    print("Number of parameters : ", num_parameters)
    print("Input Shape : ", x.shape)
    print("Output Shape : ", y.shape)
    print("Output : ", y)
    
    
class Block(nn.Module):
    """Simple Block that does some linear transformation followed by a ReLU nonlinearity"""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim)/math.sqrt(dim)) #    Wraps it as an nn.Parameter, so PyTorch registers it as a learnable model parameter
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x@self.weight
        x = F.relu(x)
        return x
    
    
class DeepNetwork(nn.Module) :
    """Map `dim` -vector to a `dim` -vector""" 
    def __init__(self, dim:int, num_layers:int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #Apply all the layers sequentially
        for layer in self.layers:
            #Take the input tensor x, pass it through Block 1, then take that output and pass it through Block 2, then Block 3, and so on.
            x = layer(x)
        return x
    
deep_network()
        
        