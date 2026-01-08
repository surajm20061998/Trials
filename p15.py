#dataloaders and datasets with pytorch
# and Batch Training

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math 

class WineDataset(Dataset):
    
    def __init__(self, transform=None):
        #data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype = np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:,1:]
        self.y = xy[:, [0]]
        
        self.transform = transform
        
        
        
        
    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def __len__(self):
        #len(dataset)
        return self.n_samples
    

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform():
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
        
    
dataset = WineDataset(transform=ToTensor())

first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)



composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset2 = WineDataset(transform=composed)
first_data2 = dataset2[0]
features2, labels2 = first_data2
print(type(features2), type(labels2))
print(features2, labels2)

# dataLoader = DataLoader(dataset = dataset, batch_size=4, shuffle=True, num_workers=0)

# dataIter = iter(dataLoader)
# data = next(dataIter)
# features, labels = data
# print(features, labels)


# # dummy training loop
# num_epochs = 2
# total_samples = len(dataset)
# n_iter = math.ceil(total_samples/4)
# print(total_samples, n_iter)

# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataLoader):
#         #forward
        
#         #backward
        
#         #update weights
        
#         if (i+1)%5==0:
#             print(f'epoch : {epoch+1}/{num_epochs}, step : {i+1}/{n_iter}, inputs : {inputs.shape}')