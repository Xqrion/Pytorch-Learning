from PIL import Image
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

img = Image.open("Images/星野爱.jpg")
writer = SummaryWriter("logs")

#ToTensor 的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("ToTensor_img", img_tensor, global_step=1)

#Normalize 的作用
trans_normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])#RGB 三个
img_normalize = trans_normalize(img_tensor)

writer.add_image("Normalize_img", img_normalize, global_step=1)

#Resize
print(img.size)
trans_resize = transforms.Resize((224, 224))
img_resize = trans_resize(img)#这里返回值是PIL,输入也是PIL
#PIL -> resize -> PIL
img_resize = trans_totensor(img_resize)
writer.add_image("Resize_img", img_resize, global_step=1)

trans_resize_2 = transforms.Resize(224)
img_resize_2 = trans_resize_2(img)
print(img_resize_2.size)
img_resize_2 = trans_totensor(img_resize_2)
writer.add_image("Resize_img", img_resize_2, global_step=2)


#Compose

trans_compose = transforms.Compose([transforms.Resize(512), trans_totensor])
#版本更新后，传tensor也可以了
img_compose = trans_compose(img)

writer.add_image("Compose_img", img_compose, global_step=1)

#RandomCrop
trans_randomcrop = transforms.RandomCrop(224)

for i in range(10):
    img_crop = trans_randomcrop(img_tensor)
    writer.add_image("RandomCrop_img", img_crop, global_step=i)

#print(type(...))

writer.close()