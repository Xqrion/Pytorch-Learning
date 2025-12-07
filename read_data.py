from torch.utils.data import Dataset
from PIL import Image
import os

#自定义数据集
class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir, img_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir #这里的label是相对于rootdir的
        self.img_dir = img_dir

        #self.path = os.path.join(root_dir, self.label_dir)
        self.img_path = os.listdir(os.path.join(root_dir, self.img_dir))
        self.label_path = os.listdir(os.path.join(root_dir, self.label_dir))

    def __getitem__(self, index):
        img_name = self.img_path[index]#图片的名字
        label_name = self.label_path[index]
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)#图片的相对路径
        img = Image.open(img_item_path)

        #label_path = os.path.join(self.root_dir, self.label_dir)
        #label = os.read(os.listdir(label_path)[index])
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        with open(label_item_path, 'r') as f:
            label = f.readlines()
        return img, label

    def __len__(self):
        return len(self.img_path)



root_dir = "dataset/hymenoptera_data/train"
ant_label_dir = "ants_label"
ant_img_dir = "ants_image"

ants_dataset = MyDataset(root_dir, ant_label_dir, ant_img_dir)

print(ants_dataset[1])
