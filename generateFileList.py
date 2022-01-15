import cv2
import os
import pickle
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split
from IgAModel66.setting import config


# 添加文件名单到 result/csv里

files=os.listdir(config['server_path']+'IgAModel/test/M0/')
files.extend(os.listdir(config['server_path']+'IgAModel/test/M1/') )


eval_All=pd.read_csv('result/eval_All_0.73.csv',header=0)


eval_All['file']=files


eval_All.to_csv('result/eval_All_0.73_file.csv',index=False)