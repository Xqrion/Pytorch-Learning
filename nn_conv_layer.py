import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader import writer

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        #这里是卷积层(class)，不是F里面的卷积操作

    def forward(self, x):
        x = self.conv1(x)
        return x


mynn = MyNN()
print(mynn)

writer = SummaryWriter("logs/mynn_conv2D")
step = 0
for data in dataloader:
    imgs, targets = data
    output = mynn(imgs)
    print(output.shape)

    #shape (64, 3, 32, 32)
    writer.add_images("input", imgs, step)
    #shape (64, 6, 32, 32) 这里channel 不对， 彩色图像是三个
    #writer.add_images("output", output, step)

    #如果用reshape 就会把channel 转移到batchsize里
    output = torch.reshape(output, (-1, 1, 30, 30))
    writer.add_images("output", output, step)

    step += 1

writer.close()