from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)
writer.add_image("test img", img_array, 2, dataformats="HWC")
writer.close()

for i in range(0,100):
    writer.add_scalar("y=2 x=1 test",i, i)

writer.close()