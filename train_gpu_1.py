import time

##利用GPU训练的模式1 : 调用CUDA 模型、数据、损失函数
#完整的训练流程
import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from model import *
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


#准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size, test_data_size)

#DataLoader 加载
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

#搭建神经网路
mynn = XqrionNet()
mynn = mynn.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

#优化器
learning_rate = 0.01#1e-2
optimizer = torch.optim.SGD(mynn.parameters(), lr=learning_rate)


#设置训练网路的参数
total_train_steps = 0

total_test_steps = 0

epoch = 10#循环轮数

#添加tensorboard
writer = SummaryWriter("./logs/train")

start_time = time.time()

mynn.train()#在一些特殊的层上
for i in range(epoch):
    print('epoch:', i+1, '/', epoch)
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = mynn(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps += 1
        if (total_train_steps) % 100 == 0:
            print("loss:", loss.item(), "训练次数:", total_train_steps)
            writer.add_scalar("loss", loss.item(), total_train_steps)
            end_time = time.time()

            print(end_time - start_time, "时间花费")

    #train数据训练了一遍
    #下面测试
    mynn.eval()#在一些特殊层上

    total_test_loss = 0
    total_accuracy_number = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = mynn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            #计算整体正确的个数 [十个类别的索引]
            accuracy_number = (outputs.argmax(1) == targets).sum().item()
            total_accuracy_number += accuracy_number
    print("整体的loss:", total_test_loss)
    print("accuracy:", total_accuracy_number/test_data_size)
    writer.add_scalar("accuracy", total_accuracy_number/test_data_size, total_test_steps)
    writer.add_scalar("loss_test", total_test_loss, total_test_steps)
    total_test_steps += 1

    #保存每一轮的结果
    torch.save(mynn , "models/xqrion_{}.pth".format(i))
    print("第{}轮的模型保存了".format(i))







