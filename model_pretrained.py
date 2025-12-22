import torchvision.datasets
from torch import nn
from torchvision import transforms
from torchvision.models import VGG16_Weights

#train_data = torchvision.datasets.ImageNet(root='./data', train=True, download=True, transform=transforms.ToTensor())


vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

print(vgg16_true)

#vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)#直接修改
