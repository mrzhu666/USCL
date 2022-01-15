import os
import pandas as pd
import numpy as np


result=pd.read_csv('result/All.csv',header=0)

# print(result['All'].iloc[-1])



paraLog=pd.DataFrame(columns={'LR','weight_decay','L'})
paraLog=paraLog.append({'LR':1,'weight_decay':1},ignore_index=True)
print(((paraLog['LR']==1) & (paraLog['weight_decay']==2)).any())

paraLog.to_csv('ss.csv',index=False)

paraLog=pd.read_csv('ss.csv')
print(paraLog.head())