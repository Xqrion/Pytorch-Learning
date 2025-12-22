#这个test不是临时文件，而是训练好并验证好了后真正投入使用测试的
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from model import *

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

img_path  = "./Images/deer.png"

image = Image.open(img_path)
image = image.convert("RGB")
transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
image = transform(image)


print(image.shape)

model = torch.load("./models/gpu/xqrion_gpu2_29.pth",
                   weights_only=False,map_location=torch.device('cuda'))#没有cuda改成cpu
image = torch.reshape(image, (1,3,32,32))


model.eval()
with torch.no_grad():
    output = model(image.to("cuda"))
print(output.argmax())
print(output)
print(train_data.classes[output.argmax().item()])
print(train_data.class_to_idx)
