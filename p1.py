import torch

#x = torch.empty(2, 3, 2)

# x = torch.rand(2, 2, 2, dtype=torch.float16)
# y = torch.rand(2,2,2, dtype=torch.float16)
# #z = x+y
# z = torch.add(x,y)

# print(x.dtype)
# print(x.type())
# print(x.size())

# print(y.dtype)
# print(y.type())
# print(y.size())

# print(z.dtype)
# print(z.type())
# print(z.size())

# print(x)
# print(y)
# print(z)



x = torch.rand(5,5)
print(x)
print(x[:,0])
print(x[0,:])
print(x[1,1])
print(x[1,1].item())


y = torch.rand(4,4)
print(y)
z = y.view(2,8)
print(z)