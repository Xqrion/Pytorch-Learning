from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# tensor hen zhong yao

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_transform = transforms.ToTensor()
tensor_img = tensor_transform(img)

writer.add_image("img", tensor_img)
writer.close()