#完整的训练流程
import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import *

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

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate = 0.01#1e-2
optimizer = torch.optim.SGD(mynn.parameters(), lr=learning_rate)

#设置训练网路的参数
total_train_steps = 0

total_test_steps = 0

epoch = 10#循环轮数

#添加tensorboard
writer = SummaryWriter("./logs/train")

mynn.train()#在一些特殊的层上
for i in range(epoch):
    print('epoch:', i+1, '/', epoch)
    for data in train_loader:
        imgs, targets = data
        outputs = mynn(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps += 1
        if (total_train_steps) % 100 == 0:
            print("loss:", loss.item(), "训练次数:", total_train_steps)
            writer.add_scalar("loss", loss.item(), total_train_steps)

    #train数据训练了一遍
    #下面测试
    mynn.eval()#在一些特殊层上

    total_test_loss = 0
    total_accuracy_number = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
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
    torch.save(mynn , "model/xqrion_{}.pth".format(i))
    print("第{}轮的模型保存了".format(i))







