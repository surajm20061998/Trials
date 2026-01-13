# #Saving and loading models

# import torch
# import torch.nn as nn 

# #way1
# torch.save(arg, PATH)

# torch.load(PATH)


# model.load_state_dict(arg)
# model.eval()



# #way2
# torch.save(model.state_dict(), PATH)

# #load
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

import torch
import torch.nn as nn 
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features,1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model(n_input_features=6)

#train model
criterion = nn.BCELoss()                 # binary classification loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# dummy dataset
X = torch.randn(100, 6)                  # 100 samples, 6 features
y = torch.randint(0, 2, (100, 1)).float()  # binary labels

# -----------------------
# training loop
# -----------------------
num_epochs = 100

for epoch in range(num_epochs):
    # forward
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward
    optimizer.zero_grad()
    loss.backward()

    # update weights & bias
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# -----------------------
# check learned parameters
# -----------------------
print("\nLearned weights:", model.linear.weight.data)
print("Learned bias:", model.linear.bias.data)


#lazy method
FILE = "model1.pth"
torch.save(model, FILE)


#Correct Method
# FILE = "model1.pth"
# torch.save(model.state_dict(), FILE)

