#using pytorch lightnig

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

import pytorch_lightning as pl 


#hyper parameters
input_size = 784 #images are 28x28
hidden_size = 500 #can try out more
num_classes = 10
num_epochs = 10
batch_size = 100
leraning_rate = 0.001


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        #Since we use the cross entropy loss, it applies the softmax for us hence we dont use it here   
        return out


