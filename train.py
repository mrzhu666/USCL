
import re
from numpy import log
import yaml
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 命令行控制训练,保证无关变量一致
# 模型结构定型后，调参前需要修改的文件
# eval_pretrained_model.py 日记文件：用参数控制是否记录
# setting.py  参数文件路径：改用绝对路径定位
# 防止参数重复?
# 参考最近运行的数据？

lr_range=range(5,100,5)
weight_range=range(1,250,5)
iter=[(lr,weight) for lr in lr_range for weight in weight_range]

def paralog():
    if not os.path.exists(log_path+'param.csv'):
        return pd.DataFrame(columns={'LR','weight_decay'})
    return pd.read_csv(log_path+'param.csv',header=0)

log_path='log/IgAModel66All/'
w=SummaryWriter(log_path)

for lr,weight in tqdm(iter):
    print('lr',lr,'weight',weight)
    paraLog=paralog()
    with open('./IgAModel66/config.yaml', encoding='utf-8') as f:
        config=yaml.safe_load(f)

    config['param']['LR']=lr/100000
    config['param']['weight_decay']=weight/10000
    paramDict={'LR':config['param']['LR'],'weight_decay':config['param']['weight_decay']}
    print(paramDict)
    # 重复参数过滤
    if ((paraLog['LR']==config['param']['LR']) & (paraLog['weight_decay']==config['param']['weight_decay'])).any():
        print(paramDict)
        print('参数跳过')
        continue

    with open('./IgAModel66/config.yaml', "w", encoding="utf-8") as f:
        yaml.dump(config,f,allow_unicode=True)
    
    # result=os.system('python ./IgAModel66/eval_pretrained_model.py')
    result=os.system('python ./IgAModel66/eval_All_data.py')
    print('result ',result)
    if result==2:  # 键盘中断
        break
    # 键盘终端不一定能检测到,通过文件检测是否有产生结果
    if os.path.exists('result/All.csv'):
        resultCsv=pd.read_csv('result/All.csv',header=0)
        acc=resultCsv['All'].iloc[-1]
        w.add_hparams(paramDict,{'acc':acc})

        # 保存参数到csv
        paraLog=paraLog.append(paramDict,ignore_index=True)
        paraLog.to_csv(log_path+'param.csv',index=False)
        os.remove('result/All.csv')

w.close()

print('调参完成')