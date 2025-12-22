import torch
import torchvision

#保存方式1加载模型

model = torch.load("./models/vgg_method1.pth", weights_only=False)

#保存方式2

vgg16 = torchvision.models.vgg16(weights=None)
model = torch.load("./models/vgg_method2.pth", weights_only=False)
vgg16.load_state_dict(model)



print(model)
