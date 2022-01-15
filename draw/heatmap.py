import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




plt.figure()


# 混淆矩阵
confusion_matrix=[[60.,40.],
                  [25.,75.]]
# confusion_matrix=[[26.,24.],
#                   [3.,47.]]
# classes=len(confusion_matrix)   
# for i in range(classes):
#     classSum=sum(confusion_matrix[i])
#     for j in range(classes):
#         confusion_matrix[i][j]/=classSum

confusion=pd.DataFrame({'Normal':confusion_matrix[:][0],'Abnormal':confusion_matrix[:][1]})
confusion.index=('Normal','Abnormal')
print(confusion)

cmap=sns.color_palette("Reds",as_cmap=True)
# sns.heatmap(confusion_matrix, annot = True, cmap=cmap,vmin=0,vmax=1)
sns.heatmap(confusion, annot = True, cmap=cmap)

plt.xlabel('Predictions',fontsize = 19)
plt.ylabel('Truth',fontsize = 19)

# plt.title('Histogram of the Dataset', fontsize = 19)
# plt.savefig("figure/HeatMap.png",dpi=200)
plt.savefig("figure/HeatMap.png",dpi=200)