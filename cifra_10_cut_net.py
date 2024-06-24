import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
torch.set_printoptions(threshold=np.inf)
import torch
from torchsummary import summary
use_cuda = torch.cuda.is_available()
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torchsummary import summary

#---------------------------------------------------------------------------------------
def cut_conv1_pw_cnt():
    global conv1_pw_cnt
    conv1_pw_cnt += 1
    return conv1_pw_cnt
conv1_pw_cnt = 0
def cut_conv1_pw_all():
    global conv1_pw_all
    conv1_pw_all += 1
    return conv1_pw_all
conv1_pw_all = 0
def cut_conv1_pw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((max_obs ==0)):
            #全零剪枝
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
                x[m][k] = 0
                cut_conv1_pw_cnt()
            cut_conv1_pw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv2_dw_cnt():
    global conv2_dw_cnt
    conv2_dw_cnt += 1
    return conv2_dw_cnt
conv2_dw_cnt = 0
def cut_conv2_dw_all():
    global conv2_dw_all
    conv2_dw_all += 1
    return conv2_dw_all
conv2_dw_all = 0
def cut_conv2_dw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if (float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.15):#全零剪枝
                x[m][k] = 0
                cut_conv2_dw_cnt()
            cut_conv2_dw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv2_pw_cnt():
    global conv2_pw_cnt
    conv2_pw_cnt += 1
    return conv2_pw_cnt
conv2_pw_cnt = 0
def cut_conv2_pw_all():
    global conv2_pw_all
    conv2_pw_all += 1
    return conv2_pw_all
conv2_pw_all = 0
def cut_conv2_pw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if ((max_obs ==0)):
                x[m][k] = 0
                cut_conv2_pw_cnt()
            cut_conv2_pw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv3_dw_cnt():
    global conv3_dw_cnt
    conv3_dw_cnt += 1
    return conv3_dw_cnt
conv3_dw_cnt = 0
def cut_conv3_dw_all():
    global conv3_dw_all
    conv3_dw_all += 1
    return conv3_dw_all
conv3_dw_all = 0
def cut_conv3_dw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if (float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.15):#全零剪枝
                x[m][k] = 0
                cut_conv3_dw_cnt()
            cut_conv3_dw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv3_pw_cnt():
    global conv3_pw_cnt
    conv3_pw_cnt += 1
    return conv3_pw_cnt
conv3_pw_cnt = 0
def cut_conv3_pw_all():
    global conv3_pw_all
    conv3_pw_all += 1
    return conv3_pw_all
conv3_pw_all = 0
def cut_conv3_pw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if ((max_obs ==0)):
                x[m][k] = 0
                cut_conv3_pw_cnt()
            cut_conv3_pw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv4_dw_cnt():
    global conv4_dw_cnt
    conv4_dw_cnt += 1
    return conv4_dw_cnt
conv4_dw_cnt = 0
def cut_conv4_dw_all():
    global conv4_dw_all
    conv4_dw_all += 1
    return conv4_dw_all
conv4_dw_all = 0
def cut_conv4_dw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if (float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.15):#全零剪枝
                x[m][k] = 0
                cut_conv4_dw_cnt()
            cut_conv4_dw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv4_pw_cnt():
    global conv4_pw_cnt
    conv4_pw_cnt += 1
    return conv4_pw_cnt
conv4_pw_cnt = 0
def cut_conv4_pw_all():
    global conv4_pw_all
    conv4_pw_all += 1
    return conv4_pw_all
conv4_pw_all = 0
def cut_conv4_pw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if ((max_obs ==0)):
                x[m][k] = 0
                cut_conv4_pw_cnt()
            cut_conv4_pw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv5_dw_cnt():
    global conv5_dw_cnt
    conv5_dw_cnt += 1
    return conv5_dw_cnt
conv5_dw_cnt = 0
def cut_conv5_dw_all():
    global conv5_dw_all
    conv5_dw_all += 1
    return conv5_dw_all
conv5_dw_all = 0
def cut_conv5_dw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if (float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.15):#全零剪枝
                x[m][k] = 0
                cut_conv5_dw_cnt()
            cut_conv5_dw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv5_pw_cnt():
    global conv5_pw_cnt
    conv5_pw_cnt += 1
    return conv5_pw_cnt
conv5_pw_cnt = 0
def cut_conv5_pw_all():
    global conv5_pw_all
    conv5_pw_all += 1
    return conv5_pw_all
conv5_pw_all = 0
def cut_conv5_pw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if ((max_obs ==0)):
                x[m][k] = 0
                cut_conv5_pw_cnt()
            cut_conv5_pw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv6_dw_cnt():
    global conv6_dw_cnt
    conv6_dw_cnt += 1
    return conv6_dw_cnt
conv6_dw_cnt = 0
def cut_conv6_dw_all():
    global conv6_dw_all
    conv6_dw_all += 1
    return conv6_dw_all
conv6_dw_all = 0
def cut_conv6_dw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if (float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.15):#全零剪枝
                x[m][k] = 0
                cut_conv6_dw_cnt()
            cut_conv6_dw_all()
    return x
#---------------------------------------------------------------------------------------
def cut_conv6_pw_cnt():
    global conv6_pw_cnt
    conv6_pw_cnt += 1
    return conv6_pw_cnt
conv6_pw_cnt = 0
def cut_conv6_pw_all():
    global conv6_pw_all
    conv6_pw_all += 1
    return conv6_pw_all
conv6_pw_all = 0
def cut_conv6_pw(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.03)&(float(max_obs)<8)):
            if ((max_obs ==0)):
                x[m][k] = 0
                cut_conv6_pw_cnt()
            cut_conv6_pw_all()
    return x
class tinymcunet(nn.Module):

    def __init__(self, in_channels=3, num_classes=10, **kwargs):
        super(tinymcunet, self).__init__()
        # 开始的一个卷积快用于映射特征
        self.conv1_pw = Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 1), stride=1, padding=0)# 64*64=4096=4kB
        self.conv1_pw_BatchNorm2d = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2_dw = Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), groups=16, stride=1, padding=1)#64*64=4096=4kB
        self.conv2_dw_BatchNorm2d = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.conv2_pw = Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=1, padding=0)# 64*64=4096=4kB
        self.conv2_pw_BatchNorm2d = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.maxpool1 = MaxPool2d(2, 2)

        self.conv3_dw = Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32,stride=1, padding=1)  # 32*32=1024=1kB
        self.conv3_dw_BatchNorm2d = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv3_pw = Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=1, padding=0)  # 32*32=1024=1kB
        self.conv3_pw_BatchNorm2d = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.maxpool2 = MaxPool2d(2, 2)

        self.conv4_dw = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64,stride=1, padding=1)  # 16
        self.conv4_dw_BatchNorm2d = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv4_pw = Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)  # 16
        self.conv4_pw_BatchNorm2d = nn.BatchNorm2d(128)
        self.relu7 = nn.ReLU()

        self.maxpool3 = MaxPool2d(2, 2)

        self.conv5_dw = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), groups=128,stride=1, padding=1)  #8
        self.conv5_dw_BatchNorm2d = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.conv5_pw = Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0) #8
        self.conv5_pw_BatchNorm2d = nn.BatchNorm2d(256)
        self.relu9 = nn.ReLU()

        self.maxpool4 = MaxPool2d(2, 2)

        self.conv6_dw = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), groups=256,stride=1, padding=1)  #4
        self.conv6_dw_BatchNorm2d = nn.BatchNorm2d(256)
        self.relu10 = nn.ReLU()

        self.conv6_pw = Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=1, padding=0) #4
        self.conv6_pw_BatchNorm2d = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU()
        self.maxpool5 = MaxPool2d(2, 2)
        # self.maxpool5 = nn.AvgPool2d(2, 2)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512*4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = quan_conv(x)
        x = self.conv1_pw(x)
        x = self.conv1_pw_BatchNorm2d(x)
        x = self.relu1(x)
        # print(x)

        x = quan_conv(x)
        x = cut_conv1_pw(x)
        print('conv1_pw_cut_all:', conv1_pw_cnt, 'conv1_pw_all:', conv1_pw_all, 'conv1_cut_percent:', conv1_pw_cnt / conv1_pw_all)
        # print(x)

        x = self.conv2_dw(x)
        x = self.conv2_dw_BatchNorm2d(x)
        x = self.relu2(x)


        x = quan_conv(x)
        x = cut_conv2_dw(x)
        print('conv2_dw_cut_all:', conv2_dw_cnt, 'conv2_dw_all:', conv2_dw_all, 'conv2_dw_cut_percent:', conv2_dw_cnt / conv2_dw_all)
        # print(x)

        x = self.conv2_pw(x)
        x = self.conv2_pw_BatchNorm2d(x)
        x = self.relu3(x)

        x = quan_conv(x)
        x = cut_conv2_pw(x)
        print('conv2_pw_cut_all:', conv2_pw_cnt, 'conv2_pw_all:', conv2_pw_all, 'conv2_pw_cut_percent:', conv2_pw_cnt / conv2_pw_all)
        # print(x)

        x = self.maxpool1(x)
        x = self.conv3_dw(x)
        x = self.conv3_dw_BatchNorm2d(x)
        x = self.relu4(x)

        x = quan_conv(x)
        x = cut_conv3_dw(x)
        print('conv3_dw_cut_all:', conv3_dw_cnt, 'conv3_dw_all:', conv3_dw_all, 'conv3_dw_cut_percent:', conv3_dw_cnt / conv3_dw_all)
        # print(x)
        x = self.conv3_pw(x)
        x = self.conv3_pw_BatchNorm2d(x)
        x = self.relu5(x)

        x = quan_conv(x)
        x = cut_conv3_pw(x)
        print('conv3_pw_cut_all:', conv3_pw_cnt, 'conv3_pw_all:', conv3_pw_all, 'conv3_pw_cut_percent:', conv3_pw_cnt / conv3_pw_all)
        # print(x)
        x = self.maxpool2(x)
        x = self.conv4_dw(x)
        x = self.conv4_dw_BatchNorm2d(x)
        x = self.relu6(x)
        # print(x)
        x = quan_conv(x)
        x = cut_conv4_dw(x)
        print('conv4_dw_cut_all:', conv4_dw_cnt, 'conv4_dw_all:', conv4_dw_all, 'conv4_dw_cut_percent:', conv4_dw_cnt / conv4_dw_all)
        # print(x)
        x = self.conv4_pw(x)
        x = self.conv4_pw_BatchNorm2d(x)
        x = self.relu7(x)
        # print(x)
        x = quan_conv(x)
        x = cut_conv4_pw(x)
        print('conv4_pw_cut_all:', conv4_pw_cnt, 'conv4_pw_all:', conv4_pw_all, 'conv4_pw_cut_percent:', conv4_pw_cnt / conv4_pw_all)
        # print(x)
        x = self.maxpool3(x)
        x = self.conv5_dw(x)
        x = self.conv5_dw_BatchNorm2d(x)
        x = self.relu8(x)
        # print(x)
        x = quan_conv(x)
        x = cut_conv5_dw(x)
        print('conv5_dw_cut_all:', conv5_dw_cnt, 'conv5_dw_all:', conv5_dw_all, 'conv5_dw_cut_percent:', conv5_dw_cnt / conv5_dw_all)
        # print(x)
        x = self.conv5_pw(x)
        x = self.conv5_pw_BatchNorm2d(x)
        x = self.relu9(x)

        x = quan_conv(x)
        x = cut_conv5_pw(x)
        print('conv5_pw_cut_all:', conv5_pw_cnt, 'conv5_pw_all:', conv5_pw_all, 'conv5_pw_cut_percent:', conv5_pw_cnt / conv5_pw_all)
        # print(x)
        x = self.maxpool4(x)
        x = self.conv6_dw(x)
        x = self.conv6_dw_BatchNorm2d(x)
        x = self.relu10(x)

        x = quan_conv(x)
        x = cut_conv6_dw(x)
        print('conv6_dw_cut_all:', conv6_dw_cnt, 'conv6_dw_all:', conv6_dw_all, 'conv6_dw_cut_percent:', conv6_dw_cnt / conv6_dw_all)
        # print(x)
        x = self.conv6_pw(x)
        x = self.conv6_pw_BatchNorm2d(x)
        x = self.relu11(x)

        x = quan_conv(x)
        x = cut_conv6_pw(x)
        print('conv6_pw_cut_all:', conv6_pw_cnt, 'conv6_pw_all:', conv6_pw_all, 'conv6_pw_cut_percent:', conv6_pw_cnt / conv6_pw_all)
        # print(x)
        x = self.maxpool5(x)
        # 将三维图像展平为二维分类特征
        x = torch.flatten(x, 1)

        # x = quan_fc(x)
        # x = self.dropout(x)
        x = self.fc(x)
        return x

def quan_conv1(x):
    zero_num_all = 0
    for m in range(len(x)):
        for k in range(len(x[0])):
            min_ci_fang = 0
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            #是否全为0
            if(torch.eq(max_obs, 0)==0):
                # print(max_obs,'Channel_all:',len(x[0]),'Channel_num:',k,'H:',len(x[0][0]),'W:',len(x[0][0][0]))
                while max_obs > 1 / 4096:
                    max_obs >>= 1
                    min_ci_fang += 1
                x[m][k] = quantify_tensor(x[m][k], min_ci_fang)
            else:
                zero_num_all        =   zero_num_all    +   1
    return x
def quan_conv(x):
    zero_num_all = 0
    for m in range(len(x)):
        for k in range(len(x[0])):
            min_ci_fang = 0
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            ''''''
            #是否全为0
            if(torch.eq(max_obs, 0)==0):
                # print(max_obs,'Channel_all:',len(x[0]),'Channel_num:',k,'H:',len(x[0][0]),'W:',len(x[0][0][0]))
                while max_obs > 1 / 4096:
                    max_obs >>= 1
                    min_ci_fang += 1
                x[m][k] = quantify_tensor(x[m][k], min_ci_fang)
            else:
                zero_num_all = zero_num_all + 1
                # print('Total num:',len(x[0][0])*len(x[0][0][0]),'None Zero num:',torch.count_nonzero(x[m][k]).item(),'None Zero Percent:',float(torch.count_nonzero(x[m][k]).item())/(len(x[0][0])*len(x[0][0][0])),'Channel_all:',len(x[0]),'Channel_num:',k,'H:',len(x[0][0]),'W:',len(x[0][0][0]))  # 返回tensor中不为0的数据个数

            '''
            #是否小于阈值
            a = torch.tensor([0.5])
            if(torch.less(max_obs, a)):
                print(max_obs,'Channel_all:',len(x[0]),'Channel_num:',k,'H:',len(x[0][0]),'W:',len(x[0][0][0]))
            '''
            '''
            #稀疏程度
            if(float(torch.count_nonzero(x[m][k]).item())/(len(x[0][0])*len(x[0][0][0]))<0.1):
                print('Total num:',len(x[0][0])*len(x[0][0][0]),'None Zero num:',torch.count_nonzero(x[m][k]).item(),'None Zero Percent:',float(torch.count_nonzero(x[m][k]).item())/(len(x[0][0])*len(x[0][0][0])),'Channel_all:',len(x[0]),'Channel_num:',k,'H:',len(x[0][0]),'W:',len(x[0][0][0]))  # 返回tensor中不为0的数据个数
            '''
            # while max_obs > 1 / 4096:
            #     max_obs >>= 1
            #     min_ci_fang += 1
            # x[m][k]= quantify_tensor(x[m][k], min_ci_fang)
            # print("Range:", pow(2, min_ci_fang - 12), "Min_unit:", pow(2, min_ci_fang - 19))
            # print(x[m][k])
    # print('Zero num',zero_num_all,'All num',len(x[0]),'Zero percent:',100*float(zero_num_all) / len(x[0]),'%')
    return x
def quan_fc(x):
    for i in range(len(x)):
        min_ci_fang = 0
        max = torch.max(x[i])
        min = torch.min(x[i])
        max_obs = torch.max(max, (-min))
        while max_obs > 1 / 4096:
            max_obs >>= 1
            min_ci_fang += 1
        for j in range(len(x[0])):
            x[i][j] = quantify_data(x[i][j], min_ci_fang)
    return x
def quantify_data(input_data,min_ci_fang):
    output = 0
    for i in range(8):
        if (input_data >= pow(2, min_ci_fang - 12 - i)):
            input_data = input_data - pow(2, min_ci_fang - 12 - i)
            output = output + pow(2, min_ci_fang - 12 - i)
    return output
def quantify_tensor(input_data,min_ci_fang):
    # print(input_data)
    input_data_shift = input_data << (19 - min_ci_fang)
    output = torch.round(input_data_shift)>>(19 - min_ci_fang)
    # print(output)
    return output