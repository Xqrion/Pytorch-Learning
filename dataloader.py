import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset_transform import writer

test_data = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, drop_last=False)
#shuffle 打乱 drop_last舍弃最后一个

img, target = test_data[0]
#dataset的使用

writer = SummaryWriter(log_dir="logs/dataloader")

step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()