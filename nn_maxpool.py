import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader import writer

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

input = torch.reshape(input, (1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        output = self.max_pool1(x)
        return output

mynn = Mynn()
# output = mynn(input)
# print(output)
# ceil_mode = True =>算了边缘进去
# ([[[[2, 3],
#           [5, 1]]]])

# ceil_mode = False =>不算了边缘进去
# [2]

writer = SummaryWriter("logs/mynn_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs, step)
    output = mynn(imgs)
    #channel 不会变
    writer.add_images("output",output, step)
    step += 1

writer.close()
