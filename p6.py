import torch

x=torch.tensor(1.0)
y=torch.tensor(2.0)
w=torch.tensor(1.0, requires_grad=True)

#Forward_Pass

y_hat = w*x
loss = (y_hat - y)**2
print(loss)

#Backward_Pass

loss.backward()
print(w.grad)


#Update weights
#Do next forward and backward pass
