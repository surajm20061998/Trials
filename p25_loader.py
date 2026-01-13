#Lazy Way
import torch
import torch.nn as nn
FILE = "model1.pth"

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = torch.load(FILE, weights_only=False)
model.eval()

for name, param in model.named_parameters():
    print(name, param.data)






#Correct way
# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, n_input_features):
#         super(Model, self).__init__()
#         self.linear = nn.Linear(n_input_features, 1)

#     def forward(self, x):
#         return torch.sigmoid(self.linear(x))

# FILE = "model1.pth"

# model = Model(n_input_features=6)
# model.load_state_dict(torch.load(FILE))   # loads weights safely
# model.eval()

# for name, param in model.named_parameters():
#     print(name, param.data)