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


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    print("Apex on, run on mixed precision.")
    apex_support = False
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nRunning on:", device)

if device == 'cuda':
    device_name = torch.cuda.get_device_name()
    print("The device name is:", device_name)
    cap = torch.cuda.get_device_capability(device=None)
    print("The capability of this device is:", cap, '\n')

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    """返回某次epoch中最佳概率"""
    # ============================ step 1/5 data ============================
    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
    ])

    # MyDataset
    train_data = COVIDDataset(data_dir=data_dir, train=True, transform=train_transform)
    valid_data = COVIDDataset(data_dir=data_dir, train=False, transform=valid_transform)

    # DataLoder
    # train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,sampler=ImbalancedDatasetSampler(train_data))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, sampler=ImbalancedDatasetSampler(train_data))
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # ============================ step 2/5 model ============================

    net = ResNetUSCL(base_model='resnet18', out_dim=256, num_classes=classes, pretrained=pretrained)
    # net = ResNetUSCL(base_model='resnet18', out_dim=classes, pretrained=pretrained)
    if pretrained:
        print('\nThe ImageNet pretrained parameters are loaded.')
    else:
        print('\nThe ImageNet pretrained parameters are not loaded.')

    if selfsup: # import pretrained model weights
        if(device=='cuda'):
            state_dict = torch.load(state_dict_path)
        else:
            state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))
        new_dict = {k: state_dict[k] for k in list(state_dict.keys())
                    if not (k.startswith('l')
                            | k.startswith('fc'))}  # # discard MLP and fc
        model_dict = net.state_dict()

        model_dict.update(new_dict)
        net.load_state_dict(model_dict)
        print('\nThe self-supervised trained parameters are loaded.\n')
    else:
        print('\nThe self-supervised trained parameters are not loaded.\n')

    # frozen all convolutional layers
    # for param in net.parameters():
    #     param.requires_grad = False

    # fine-tune last 3 layers，前面冻结
    # 后面还有一层连接层
    for name, param in net.named_parameters():
        if not name.startswith('features.7.1'):
            param.requires_grad = False
        else:
            break
    # for name, param in net.named_parameters():
    #     if not name.startswith('model'):
    #         param.requires_grad = False
    #     else:
    #         break

    # add a classifier for linear evaluation
    # num_ftrs = net.linear.in_features  # 512
    # net.linear = nn.Linear(num_ftrs, classes)  
    # net.fc = nn.Linear(classes, classes) # fc是？？？

    # net.linear = nn.Linear(num_ftrs, 3)  # 故意的？
    # net.fc = nn.Linear(3, 3)

    for name, param in net.named_parameters():
        print(name, '\t', 'requires_grad=', param.requires_grad)

    net.to(device)

    # ============================ step 3/5 loss function ============================
    criterion = nn.CrossEntropyLoss()       # choose loss function

    # ============================ step 4/5 optimizer ============================
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)      # choose optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=MAX_EPOCH, 
                                                     eta_min=0,
                                                     last_epoch=-1)     # set learning rate decay strategy

    # ============================ step 5/5 training ============================
    print('\nTraining start!\n')
    start = time.time()
    train_curve = list()
    valid_curve = list()
    max_acc = 0.
    reached = 0    # which epoch reached the max accuracy

    # the statistics of classification result: classification_results[true][pred]
    classification_results = [[0]*classes for _ in range(classes)]
    best_classification_results = None

    # ????
    if apex_support and fp16_precision:
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level='O2',
                                        keep_batchnorm_fp32=True)
                                        
    # for epoch in tqdm(range(MAX_EPOCH)):
    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            if apex_support and fp16_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # update weights
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # print training information
            loss_mean += loss.item()
            train_curve.append(loss.item())
            # if (i+1) % log_interval == 0:
            #     loss_mean = loss_mean / log_interval
            #     print("\nTraining:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            #         epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            #     loss_mean = 0.


        loss_mean = loss_mean / log_interval
        print("\nEpoch[{:0>3}/{:0>3}] Training Loss: {:.4f} Acc:{:.2%}".format(
            epoch, MAX_EPOCH, loss_mean, correct / total),end='')
        loss_mean = 0.

        learning_rate=scheduler.get_last_lr()[0]
        # print('Learning rate this epoch:', scheduler.get_last_lr()[0])
        scheduler.step()  # updata learning rate

        # validate the model
        # if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).cpu().squeeze().sum().numpy()
                for k in range(len(predicted)):
                    classification_results[labels[k]][predicted[k]] += 1    # "label" is regarded as "predicted"

                loss_val += loss.item()

            acc = correct_val / total_val
            if acc > max_acc:   # record best accuracy
                max_acc = acc
                reached = epoch
                best_classification_results = classification_results
            classification_results = [[0]*classes for _ in range(classes)]
            valid_curve.append(loss_val/valid_loader.__len__())
            print(" Valid Loss: {:.4f} Acc:{:.2%}".format(
                loss_val, acc))
            print('Learning rate this epoch:', learning_rate)
            print('----------------------------------------------------------------------------------')

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))


    print('\nThe best prediction results of the dataset:')
    for i in range(classes):
        for j in range(classes):
            print('Class %d predicted as class %d:'%(i,j), best_classification_results[i][j])
    
    for i in range(classes):
        if best_classification_results[i][i]==0:  # 防止除数为0时报错
            acc=0
            recall=0
            print('\nClass %d accuracy:'%(i,), 0)
            print('Class %d recall:'%(i,), 0)
            print('Class %d F1:'%(i,), 0)
        else:
            acc = best_classification_results[i][i] / sum(best_classification_results[j][i] for j in range(classes))
            recall = best_classification_results[i][i] / sum(best_classification_results[i])
            print('\nClass %d accuracy:'%(i,), acc)
            print('Class %d recall:'%(i,), recall)
            print('Class %d F1:'%(i,), 2 * acc * recall / (acc + recall))
    
    return best_classification_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('-p', '--path', default='checkpoint', help='folder of ckpt')
    args = parser.parse_args()

    classes=len(config['dataset']['label'])  # 类别数
    set_seed(config['param']['seed'])  # random seed

    # parameters
    MAX_EPOCH = config['param']['epoch']       # default = 100
    BATCH_SIZE = config['param']['batch_size']       # default = 32
    # LR = 0.01             # default = 0.01
    # weight_decay = 1e-5   # default = 1e-4
    LR = 0.001             # default = 0.01
    weight_decay = 5e-3   # default = 1e-4  正则化参数，防止过拟合
    log_interval = 4
    val_interval = 1

    # 论文预训练好的模型，特征提取网络
    base_path = server_path+"IgAModel/"
    state_dict_path = os.path.join(base_path, args.path, "best_model.pth")
    print('State dict path:', state_dict_path)

    fp16_precision = True
    pretrained = False
    selfsup = config['param']['selfsup']

    # save result
    save_dir = os.path.join('result')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    resultfile = save_dir + '/my_result.txt'

    if (os.path.exists(state_dict_path)):
        confusion_matrix = np.zeros((classes,classes))
        # 五折训练
        for i in range(1, config['dataset']['fold']+1):
            print('\n' + '='*20 + 'The training of fold {} start.'.format(i) + '='*20)
            data_dir = server_path+config['dataset']['datadir'].format(i)
            best_classification_results = main()
            confusion_matrix = confusion_matrix + np.array(best_classification_results)
            print(np.array(best_classification_results))

        print('\nThe confusion matrix is:\n')
        print(confusion_matrix)
        for i in range(classes):
            print('The precision of class '+str(i)+' is:', confusion_matrix[i,i] / sum(confusion_matrix[:,i])) 

        print()
        for i in range(classes):
            print('The recall of class '+str(i)+' is:', confusion_matrix[i,i] / sum(confusion_matrix[i]))

        acc=0
        for i in range(classes):
            acc+=confusion_matrix[i,i]
        acc/=confusion_matrix.sum()
        print('\nTotal acc is:', acc)

        file_handle = open(save_dir + '/my_result.txt', mode='w+')
        for i in range(classes):
            file_handle.write('precision '+str(i)+': '+str(confusion_matrix[i,i] / sum(confusion_matrix[:,i])))
            file_handle.write('\r\n')
        for i in range(classes):
            file_handle.write('recall '+str(i)+': '+str(confusion_matrix[i,i] / sum(confusion_matrix[:,i])))
            file_handle.write('\r\n')

        file_handle.write("Total acc: "+str(acc))
        file_handle.close()