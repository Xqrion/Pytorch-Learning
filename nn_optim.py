import torch
import torchvision.datasets
from torch import nn
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding='same')#padding=2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)

        self.model = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.maxpool2,
            self.conv3,
            self.maxpool3,
            self.flatten,
            self.linear1,
            self.linear2,
        )
    def forward(self, x):
        x = self.model(x)
        return x

loss = nn.CrossEntropyLoss()

mynn = MyNN()
optimizer = torch.optim.Adam(params=mynn.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = mynn(imgs)
        result_loss = loss(outputs, targets)

        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        running_loss += result_loss
    print(running_loss)



