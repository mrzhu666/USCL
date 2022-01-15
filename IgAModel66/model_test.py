import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from setting import config
from torch.utils import data
from predict import IgA

# 测试保存模型
# IgA测试集数据
print(sys.path)

data_path=config['server_path']+'IgAModel/test/'
classes=len(config['dataset']['label']) 

confusion_matrix=[[0]*classes for _ in range(classes)]

files=os.listdir(data_path+'0/')
for file in tqdm(files):
    image=cv2.imread(data_path+'0/'+file)
    pred=IgA.predict(image)
    confusion_matrix[0][pred]+=1

files=os.listdir(data_path+'1/')
for file in files:
    image=cv2.imread(data_path+'1/'+file)
    pred=IgA.predict(image)
    confusion_matrix[1][pred]+=1

dataResult=[]
for i in range(classes):
    recall=confusion_matrix[i][i] / sum(confusion_matrix[i])
    precision=confusion_matrix[i][i] / sum(confusion_matrix[:][i])
    print('The recall of class '+str(i)+' is:', recall)
    print('The precision of class '+str(i)+' is:', precision)
    dataResult.extend([recall,precision,2 * precision * recall / (precision + recall)])
    print()
acc=0
for i in range(classes):
    acc+=confusion_matrix[i][i]
acc/=np.sum(confusion_matrix)

print('\nTotal acc is:', acc)
print()
print(confusion_matrix)