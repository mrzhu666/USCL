
import os
import sys
curDir=os.path.dirname(__file__)
sys.path.append(curDir)
import time
import random
import cv2
import numpy as np
from tqdm import tqdm
from setting import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tools.my_dataset import COVIDDataset
from resnet_uscl import ResNetUSCL
from torchsampler import ImbalancedDatasetSampler
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nRunning on:", device)

if device == 'cuda':
    device_name = torch.cuda.get_device_name()
    print("The device name is:", device_name)
    cap = torch.cuda.get_device_capability(device=None)
    print("The capability of this device is:", cap, '\n')

classes=len(config['dataset']['label'])  # 类别数

net = ResNetUSCL(base_model='resnet18', out_dim=256,num_classes=classes)

net.load_state_dict(torch.load(config['deploy_model_path'],map_location = torch.device(device)))
net.to(device)
net.eval()

class IGA():
    def __init__(self) -> None:
        self.transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
        ])
    
    def cropLRUD(self,image:np.ndarray,left:float,right:float,up:float,down:float)->np.ndarray:
        """四个边缘截图"""
        imageShape=image.shape
        a=int(up*imageShape[0])
        b=int((1-down)*imageShape[0])
        c=int(left*imageShape[1])
        d=int((1-right)*imageShape[1])
        image=image[a:b,c:d]
        return image

    def crop(self,image:np.ndarray)->np.ndarray:
        """根据图片分辨率，不同比例截取"""
        if(image.shape==(845, 1664, 3)):
            return self.cropLRUD(image,left=0.35,right=0.35,up=0.2,down=0.2)
        if(image.shape==(511, 1044, 3)):
            return self.cropLRUD(image,0.35,0.35,0.21,0.2)
        if(image.shape==(705, 1345, 3)):
            return self.cropLRUD(image,0.35,0.35,0.21,0.16)
        return image

    def preprocess(self:np.ndarray,data:np.ndarray)->np.ndarray:
        """图片预处理"""
        data=self.crop(data)
        data=cv2.resize(data,(224,224),cv2.INTER_CUBIC)
        return data

    def predict(self,data:np.ndarray)->np.ndarray:
        """输入图片，返回识别结果

        Parameters
        ----------
        data : np.ndarray
            shape: (224,224,3)

        Returns
        -------
        np.ndarray
            shape: (1)
        """        
        data=self.preprocess(data)
        img=Image.fromarray(data.astype('uint8')).convert('RGB')
        # data=data[np.newaxis,:,:,:]
        # img=data.transpose(0,3,2,1)
        # data_tensor=torch.from_numpy(img)
        # img=Image.fromarray(img.astype('uint8')).convert('RGB')
        data_tensor=self.transforms(img)
        data_tensor=data_tensor.unsqueeze(0)  # 增加一维
        data_tensor=data_tensor.to(device)   # 转GPU或CPU
        outputs=net(data_tensor)
        _, predicted = torch.max(outputs.data,0)
        return predicted.data.cpu().numpy()

IgA=IGA()