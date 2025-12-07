import torchvision
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

data_transform = transforms.Compose([
    transforms.ToTensor(),

])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,transform=data_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,transform=data_transform, download=True)
#train 决定的测试还是训练

img, target = test_set[0]

writer = SummaryWriter(log_dir='logs/transform_dataset')
for i in range(10):
    img, target = train_set[i]
    writer.add_image("test_set", img, i)

writer.close()



