import os
import random
import pickle
from PIL import Image
from torch.utils.data import Dataset
from setting import config
random.seed(1)

class COVIDDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        POCUS Dataset
            param data_dir: str
            param transform: torch.transform
        """
        # self.label_name = {"covid19": 0, "pneumonia": 1, "regular": 2}
        self.label_name = config['dataset']['label']
        with open(data_dir, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
        if train:
            self.X, self.y = X_train, y_train       # [N, C, H, W], [N]
        else:
            self.X, self.y = X_test, y_test         # [N, C, H, W], [N]
        self.transform = transform
    
    def __getitem__(self, index):
        # 为什么转灰度图？
        # eval_pretrained_model/Inputs还是3通道？因为transform
        img_arr = self.X[index].transpose(1,2,0)    # CHW => HWC shape: (224,224,3)
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB') # 0~255 (224,224) 转灰度图像
        label = self.y[index]

        if self.transform is not None:
            img = self.transform(img)  # 这里训练集和测试集变三通道

        return img, label  # 为什么不是数组？

    def get_labels(self):
        return self.y

    def __len__(self):
        return len(self.y)

