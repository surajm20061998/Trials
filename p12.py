#Logistic Regression

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#0 - Get Data
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target
n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#Make y a column vector
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#1 - Model
#f = wx + b, sigmoid in the end
class LogisticRegression(nn.Module):
    
    def __init__(self, n_input_feature):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_feature, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(n_features)


#2 - Loss and Optimizer
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#3 - Training Loop
num_epochs = 100000
for epoch in range(num_epochs):
    #forward pass and loss calculation
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    #backward pass
    loss.backward()
    
    #Update weights
    optimizer.step()
    
    #empty the gradients
    optimizer.zero_grad()
    
    if (epoch+1)%10 == 0:
        print(f'epoch : {epoch + 1}, loss : {loss.item():.4f}')
        
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()
    acc = y_pred_class.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
    
    
#plot