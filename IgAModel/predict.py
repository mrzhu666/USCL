import os
import sys
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tools.my_dataset import COVIDDataset
from resnet_uscl import ResNetUSCL
from setting import config
from torchsampler import ImbalancedDatasetSampler

result=os.popen('echo "$USER"')
user=result.read().strip()
server_path=config['user'][user]['server_path']


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nRunning on:", device)

if device == 'cuda':
    device_name = torch.cuda.get_device_name()
    print("The device name is:", device_name)
    cap = torch.cuda.get_device_capability(device=None)
    print("The capability of this device is:", cap, '\n')

net = ResNetUSCL(base_model='resnet18', out_dim=256)




