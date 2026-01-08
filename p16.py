#SoftMax and Cross Entropy

import torch
import torch.nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def crossEntropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss #/float(predicted.shape[0])

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("softmax numpy:", outputs)


x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print("softmax tensor:", outputs)


Y = np.array([1,0,0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = crossEntropy(Y,Y_pred_good)
l2 = crossEntropy(Y, Y_pred_bad)
print(f'Loss1 numpy = {l1:.4f}')
print(f'Loss2 numpy = {l2:.4f}')