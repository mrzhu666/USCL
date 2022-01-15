import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import csv
import sys
import pandas as pd
import numpy as np
import seaborn as sns

def ro_curve(y_pred, y_label, method_name):
    '''
        y_pred is a list of length n.  (0,1)    
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    fpr = dict()
    tpr = dict() 
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 3
    sns.lineplot(fpr[0],tpr[0],label= method_name + ' (area = %0.2f)' % roc_auc[0],lw=lw)
    # plt.plot(fpr[0], tpr[0],
        #  lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    return 

def col_pic():
    y_label = []
    y_pred = []
    plt.figure()

    # folder=[
    #     "fold1_0.775.csv",
    #     "fold2_0.675.csv",
    #     "fold3_0.575.csv",
    #     "fold4_0.675.csv",
    #     "fold5_0.675.csv"
    # ]
    folder=[
        "eval_All_0.73.csv",
    ]
    for i in range(1):
        f1 = pd.read_csv("result/"+folder[i])
        y_label=f1.loc[:,"label"].to_numpy()
        y_pred=f1.loc[:,"prob"].to_numpy()
        ro_curve(y_pred,y_label,"Fold" + str(i+1))
    sns.lineplot([0, 1], [0, 1], color='navy', linestyle='--')  # 对角

    fontsize = 14
    plt.title("AUC",fontsize=fontsize)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    # plt.xticks(font="Times New Roman",size=18,wei ght="bold")
    # plt.yticks(font="Times New Roman",size=18,weight="bold")
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    # plt.savefig("figure/ROC_FOLD" + ".png",dpi=700)
    plt.savefig("figure/ROC_All" + ".png",dpi=700)

def main():
    col_pic()
    
if __name__=="__main__":
    main() 