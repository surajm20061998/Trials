#Implementing Linear Regression

#1 - Input Size
#2 - Output Size
#3 - forward pass
#4 - Loss and Optimizer
#5 - Training Loop
    #1 - forward pass : compute prediction and loss
    #2 - backward pass : compute gradients
    #3 - Update weights
    
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#0 - Get Data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

#make Y a column vector

Y = Y.view(Y.shape[0], 1)

n_sample, n_features = X.shape

#1 - Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

#2 - Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#3 - Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    
    #backward pass
    loss.backward()
    
    #update weights
    optimizer.step()
    
    #empyty the weights - Why?
    optimizer.zero_grad()
    
    print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
    
#plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()