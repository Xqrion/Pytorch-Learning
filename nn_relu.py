import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

input = torch.tensor([[1,-0.5],
                      [-1, 3]])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

torch.reshape(input, (1, 1, 2, 2))

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.reLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

mynn = Mynn()
# output = mynn(input)
#
# print(output)

step = 0
writer = SummaryWriter(log_dir='logs/non_linear')
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)

    output = mynn.forward(imgs)
    writer.add_images('output', output, step)
    step += 1

writer.close()
