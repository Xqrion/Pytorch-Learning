import torch
from torch import nn


class XqrionNet(nn.Module):
    def __init__(self):
        super(XqrionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

#只有单独执行的时候才会，import不会
if __name__ == '__main__':
    model = XqrionNet()
    input = torch.randn(64, 3, 32, 32)
    output = model(input)
    print(output.shape)
