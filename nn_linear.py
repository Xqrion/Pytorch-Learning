import torch
import torchvision
from torch import nn
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root="./data",download=True, train=False, transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, drop_last=True)

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.linear = nn.Linear(in_features=196608, out_features=10)

    def forward(self, x):
        x = self.linear(x)
        return x


model = Mynn()

for data in dataloader:
    imgs, targets = data
    #img [64, 3, 32, 32]
    # imgs = torch.reshape(imgs, (1,1,1,-1))
    # img[1,1,1,196608]
    imgs = torch.flatten(imgs)
    output = model(imgs)
    print(output)