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


class tinymcunet(nn.Module):

    def __init__(self, in_channels=3, num_classes=10, **kwargs):
        super(tinymcunet, self).__init__()
        # 开始的一个卷积快用于映射特征
        self.conv1_pw = Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 1), stride=1, padding=0)# 64*64=4096=4kB
        self.conv1_pw_BatchNorm2d = nn.BatchNorm2d(16)

        self.conv2_dw = Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), groups=16, stride=1, padding=1)#64*64=4096=4kB
        self.conv2_dw_BatchNorm2d = nn.BatchNorm2d(16)

        self.conv2_pw = Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=1, padding=0)# 64*64=4096=4kB
        self.conv2_pw_BatchNorm2d = nn.BatchNorm2d(32)

        self.maxpool1 = MaxPool2d(2, 2)

        # self.conv2_dw_add = Conv2d(in_channels=32 out_channels=32, kernel_size=(3, 3), groups=32, stride=1, padding=1)#64*64=4096=4kB
        # self.conv2_dw_BatchNorm2d_add = nn.BatchNorm2d(32)
        #
        # self.conv2_pw_add = Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)# 64*64=4096=4kB
        # self.conv2_pw_BatchNorm2d_add = nn.BatchNorm2d(128)
        #
        # self.conv2_dw_add1 = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), groups=128, stride=1, padding=1)#64*64=4096=4kB
        # self.conv2_dw_BatchNorm2d_add1 = nn.BatchNorm2d(32)
        #
        # self.conv2_pw_add1 = Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)# 64*64=4096=4kB
        # self.conv2_pw_BatchNorm2d_add1 = nn.BatchNorm2d(128)



        self.conv3_dw = Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32,stride=1, padding=1)  # 32*32=1024=1kB
        self.conv3_dw_BatchNorm2d = nn.BatchNorm2d(32)
        self.conv3_pw = Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=1, padding=0)  # 32*32=1024=1kB
        self.conv3_pw_BatchNorm2d = nn.BatchNorm2d(64)
        self.maxpool2 = MaxPool2d(2, 2)

        self.conv4_dw = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64,stride=1, padding=1)  # 16
        self.conv4_dw_BatchNorm2d = nn.BatchNorm2d(64)
        self.conv4_pw = Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)  # 16
        self.conv4_pw_BatchNorm2d = nn.BatchNorm2d(128)
        self.maxpool3 = MaxPool2d(2, 2)

        self.conv5_dw = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), groups=128,stride=1, padding=1)  #8
        self.conv5_dw_BatchNorm2d = nn.BatchNorm2d(128)
        self.conv5_pw = Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0) #8
        self.conv5_pw_BatchNorm2d = nn.BatchNorm2d(256)
        self.maxpool4 = MaxPool2d(2, 2)

        self.conv6_dw = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), groups=256,stride=1, padding=1)  #4
        self.conv6_dw_BatchNorm2d = nn.BatchNorm2d(256)
        self.conv6_pw = Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=1, padding=0) #4
        self.conv6_pw_BatchNorm2d = nn.BatchNorm2d(512)
        self.maxpool5 = MaxPool2d(2, 2)
        # self.maxpool5 = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512*4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1_pw(x)

        x = self.conv1_pw_BatchNorm2d(x)
        x = self.relu(x)
        # print('---pw1---')
        # print(x)
        x = self.conv2_dw(x)
        x = self.conv2_dw_BatchNorm2d(x)
        x = self.relu(x)
        # print('---2dw---')
        # print(x)
        x = self.conv2_pw(x)
        x = self.conv2_pw_BatchNorm2d(x)
        x = self.relu(x)

        x = self.maxpool1(x)

        # x = self.conv2_dw_add(x)
        # # x = self.conv2_dw_BatchNorm2d_add(x)
        # x = self.relu(x)
        #
        # x = self.conv2_pw_add(x)
        # x = self.conv2_pw_BatchNorm2d_add(x)
        # x = self.relu(x)
        #
        # x = self.conv2_dw_add1(x)
        # # x = self.conv2_dw_BatchNorm2d_add1(x)
        # x = self.relu(x)
        #
        # x = self.conv2_pw_add1(x)
        # x = self.conv2_pw_BatchNorm2d_add1(x)
        # x = self.relu(x)
        # print('---2pw---')
        # print(x)
        x = self.conv3_dw(x)
        x = self.conv3_dw_BatchNorm2d(x)
        x = self.relu(x)
        # print('---3dw---')
        # print(x)
        x = self.conv3_pw(x)
        x = self.conv3_pw_BatchNorm2d(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        # print('---3pw---')
        # print(x)
        x = self.conv4_dw(x)
        x = self.conv4_dw_BatchNorm2d(x)
        x = self.relu(x)
        # print('---4dw---')
        # print(x)
        x = self.conv4_pw(x)
        x = self.conv4_pw_BatchNorm2d(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        # print('---4pw---')
        # print(x)
        x = self.conv5_dw(x)
        x = self.conv5_dw_BatchNorm2d(x)
        x = self.relu(x)
        # print('---5dw---')
        # print(x)
        x = self.conv5_pw(x)
        x = self.conv5_pw_BatchNorm2d(x)
        x = self.relu(x)

        x = self.maxpool4(x)
        # print('---5pw---')
        # print(x)
        x = self.conv6_dw(x)
        x = self.conv6_dw_BatchNorm2d(x)
        x = self.relu(x)
        # print('---6dw---')
        # print(x)
        x = self.conv6_pw(x)
        x = self.conv6_pw_BatchNorm2d(x)
        x = self.relu(x)

        x = self.maxpool5(x)
        # print('---6pw---')
        # print(x)
        # 将三维图像展平为二维分类特征
        x = torch.flatten(x, 1)
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