import torch
import torchvision
from torchvision.models import VGG16_Weights

vgg16 = torchvision.models.vgg16(weights=None)

#1 结构参数都保存
#自己写的需要导入import
torch.save(vgg16, "./models/vgg_method1.pth")

#2 只有参数
torch.save(vgg16.state_dict(), "./models/vgg_method2.pth")
