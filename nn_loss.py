import torch
from torch import nn



input = torch.tensor([1., 2, 3.5])
target = torch.tensor([1., 2, 3])

###x-y
loss_L1 = nn.L1Loss(reduction='sum')

result = loss_L1(input, target)
print(input.shape, target.shape)
print(result)

###(x-y)^2

loss_MSE = nn.MSELoss(reduction='sum')

result = loss_MSE(input, target)
print(result)

##交叉熵

x = torch.tensor([1., 2, 3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3)) # input 要求batch_size 加 class number

loss_cross_entropy = nn.CrossEntropyLoss()
result = loss_cross_entropy(x, y)
print(result)
