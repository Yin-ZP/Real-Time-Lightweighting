import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
torch.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
import torch.nn as nn
from collections import OrderedDict
#把channel变为8的整数倍
from torchsummary import summary
use_cuda = torch.cuda.is_available()

import multiprocessing
def make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#-----------------------------------------------------------------------
def cut_conv1_zero():
    global conv1_zero
    conv1_zero += 1
    return conv1_zero
conv1_zero = 0
def cut_conv1_all():
    global conv1_all
    conv1_all += 1
    return conv1_all
conv1_all = 0
def cut_conv1(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv1_zero()

            cut_conv1_all()
    # print('Zero num',zero_num_all,'All num',len(x[0]),'Zero percent:',100*float(zero_num_all) / len(x[0]),'%')
    return x
#-----------------------------------------------------------------------
def cut_conv2_1_DW_zero():
    global conv2_1_DW_zero
    conv2_1_DW_zero += 1
    return conv2_1_DW_zero
conv2_1_DW_zero = 0
def cut_conv2_1_DW_all():
    global conv2_1_DW_all
    conv2_1_DW_all += 1
    return conv2_1_DW_all
conv2_1_DW_all = 0
def cut_conv2_1_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv2_1_DW_zero()
            cut_conv2_1_DW_all()
    # print('Zero num',zero_num_all,'All num',len(x[0]),'Zero percent:',100*float(zero_num_all) / len(x[0]),'%')
    return x
def cut_conv2_1_PW_zero():
    global conv2_1_PW_zero
    conv2_1_PW_zero += 1
    return conv2_1_PW_zero
conv2_1_PW_zero = 0
def cut_conv2_1_PW_all():
    global conv2_1_PW_all
    conv2_1_PW_all += 1
    return conv2_1_PW_all
conv2_1_PW_all = 0
def cut_conv2_1_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv2_1_PW_zero()
            cut_conv2_1_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv2_2_DW_zero():
    global conv2_2_DW_zero
    conv2_2_DW_zero += 1
    return conv2_2_DW_zero
conv2_2_DW_zero = 0
def cut_conv2_2_DW_all():
    global conv2_2_DW_all
    conv2_2_DW_all += 1
    return conv2_2_DW_all
conv2_2_DW_all = 0
def cut_conv2_2_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv2_2_DW_zero()
            cut_conv2_2_DW_all()
    return x
def cut_conv2_2_PW_zero():
    global conv2_2_PW_zero
    conv2_2_PW_zero += 1
    return conv2_2_PW_zero
conv2_2_PW_zero = 0
def cut_conv2_2_PW_all():
    global conv2_2_PW_all
    conv2_2_PW_all += 1
    return conv2_2_PW_all
conv2_2_PW_all = 0
def cut_conv2_2_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv2_2_PW_zero()
            cut_conv2_2_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv2_3_DW_zero():
    global conv2_3_DW_zero
    conv2_3_DW_zero += 1
    return conv2_3_DW_zero
conv2_3_DW_zero = 0
def cut_conv2_3_DW_all():
    global conv2_3_DW_all
    conv2_3_DW_all += 1
    return conv2_3_DW_all
conv2_3_DW_all = 0
def cut_conv2_3_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv2_3_DW_zero()
            cut_conv2_3_DW_all()
    return x
def cut_conv2_3_PW_zero():
    global conv2_3_PW_zero
    conv2_3_PW_zero += 1
    return conv2_3_PW_zero
conv2_3_PW_zero = 0
def cut_conv2_3_PW_all():
    global conv2_3_PW_all
    conv2_3_PW_all += 1
    return conv2_3_PW_all
conv2_3_PW_all = 0
def cut_conv2_3_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv2_3_PW_zero()
            cut_conv2_3_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv2_4_DW_zero():
    global conv2_4_DW_zero
    conv2_4_DW_zero += 1
    return conv2_4_DW_zero
conv2_4_DW_zero = 0
def cut_conv2_4_DW_all():
    global conv2_4_DW_all
    conv2_4_DW_all += 1
    return conv2_4_DW_all
conv2_4_DW_all = 0
def cut_conv2_4_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv2_4_DW_zero()
            cut_conv2_4_DW_all()
    return x
def cut_conv2_4_PW_zero():
    global conv2_4_PW_zero
    conv2_4_PW_zero += 1
    return conv2_4_PW_zero
conv2_4_PW_zero = 0
def cut_conv2_4_PW_all():
    global conv2_4_PW_all
    conv2_4_PW_all += 1
    return conv2_4_PW_all
conv2_4_PW_all = 0
def cut_conv2_4_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv2_4_PW_zero()
            cut_conv2_4_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv2_5_DW_zero():
    global conv2_5_DW_zero
    conv2_5_DW_zero += 1
    return conv2_5_DW_zero
conv2_5_DW_zero = 0
def cut_conv2_5_DW_all():
    global conv2_5_DW_all
    conv2_5_DW_all += 1
    return conv2_5_DW_all
conv2_5_DW_all = 0
def cut_conv2_5_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv2_5_DW_zero()
            cut_conv2_5_DW_all()
    return x
def cut_conv2_5_PW_zero():
    global conv2_5_PW_zero
    conv2_5_PW_zero += 1
    return conv2_5_PW_zero
conv2_5_PW_zero = 0
def cut_conv2_5_PW_all():
    global conv2_5_PW_all
    conv2_5_PW_all += 1
    return conv2_5_PW_all
conv2_5_PW_all = 0
def cut_conv2_5_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv2_5_PW_zero()
            cut_conv2_5_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv2_6_DW_zero():
    global conv2_6_DW_zero
    conv2_6_DW_zero += 1
    return conv2_6_DW_zero
conv2_6_DW_zero = 0
def cut_conv2_6_DW_all():
    global conv2_6_DW_all
    conv2_6_DW_all += 1
    return conv2_6_DW_all
conv2_6_DW_all = 0
def cut_conv2_6_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv2_6_DW_zero()
            cut_conv2_6_DW_all()
    return x
def cut_conv2_6_PW_zero():
    global conv2_6_PW_zero
    conv2_6_PW_zero += 1
    return conv2_6_PW_zero
conv2_6_PW_zero = 0
def cut_conv2_6_PW_all():
    global conv2_6_PW_all
    conv2_6_PW_all += 1
    return conv2_6_PW_all
conv2_6_PW_all = 0
def cut_conv2_6_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv2_6_PW_zero()
            cut_conv2_6_PW_all()
    return x


#-----------------------------------------------------------------------
def cut_conv3_1_DW_zero():
    global conv3_1_DW_zero
    conv3_1_DW_zero += 1
    return conv3_1_DW_zero
conv3_1_DW_zero = 0
def cut_conv3_1_DW_all():
    global conv3_1_DW_all
    conv3_1_DW_all += 1
    return conv3_1_DW_all
conv3_1_DW_all = 0
def cut_conv3_1_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            #是否全为0
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv3_1_DW_zero()
            cut_conv3_1_DW_all()
    # print('Zero num',zero_num_all,'All num',len(x[0]),'Zero percent:',100*float(zero_num_all) / len(x[0]),'%')
    return x
def cut_conv3_1_PW_zero():
    global conv3_1_PW_zero
    conv3_1_PW_zero += 1
    return conv3_1_PW_zero
conv3_1_PW_zero = 0
def cut_conv3_1_PW_all():
    global conv3_1_PW_all
    conv3_1_PW_all += 1
    return conv3_1_PW_all
conv3_1_PW_all = 0
def cut_conv3_1_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv3_1_PW_zero()
            cut_conv3_1_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv3_2_DW_zero():
    global conv3_2_DW_zero
    conv3_2_DW_zero += 1
    return conv3_2_DW_zero
conv3_2_DW_zero = 0
def cut_conv3_2_DW_all():
    global conv3_2_DW_all
    conv3_2_DW_all += 1
    return conv3_2_DW_all
conv3_2_DW_all = 0
def cut_conv3_2_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv3_2_DW_zero()
            cut_conv3_2_DW_all()
    return x
def cut_conv3_2_PW_zero():
    global conv3_2_PW_zero
    conv3_2_PW_zero += 1
    return conv3_2_PW_zero
conv3_2_PW_zero = 0
def cut_conv3_2_PW_all():
    global conv3_2_PW_all
    conv3_2_PW_all += 1
    return conv3_2_PW_all
conv3_2_PW_all = 0
def cut_conv3_2_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv3_2_PW_zero()
            cut_conv3_2_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv3_3_DW_zero():
    global conv3_3_DW_zero
    conv3_3_DW_zero += 1
    return conv3_3_DW_zero
conv3_3_DW_zero = 0
def cut_conv3_3_DW_all():
    global conv3_3_DW_all
    conv3_3_DW_all += 1
    return conv3_3_DW_all
conv3_3_DW_all = 0
def cut_conv3_3_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv3_3_DW_zero()
            cut_conv3_3_DW_all()
    return x
def cut_conv3_3_PW_zero():
    global conv3_3_PW_zero
    conv3_3_PW_zero += 1
    return conv3_3_PW_zero
conv3_3_PW_zero = 0
def cut_conv3_3_PW_all():
    global conv3_3_PW_all
    conv3_3_PW_all += 1
    return conv3_3_PW_all
conv3_3_PW_all = 0
def cut_conv3_3_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv3_3_PW_zero()
            cut_conv3_3_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv3_4_DW_zero():
    global conv3_4_DW_zero
    conv3_4_DW_zero += 1
    return conv3_4_DW_zero
conv3_4_DW_zero = 0
def cut_conv3_4_DW_all():
    global conv3_4_DW_all
    conv3_4_DW_all += 1
    return conv3_4_DW_all
conv3_4_DW_all = 0
def cut_conv3_4_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv3_4_DW_zero()
            cut_conv3_4_DW_all()
    return x
def cut_conv3_4_PW_zero():
    global conv3_4_PW_zero
    conv3_4_PW_zero += 1
    return conv3_4_PW_zero
conv3_4_PW_zero = 0
def cut_conv3_4_PW_all():
    global conv3_4_PW_all
    conv3_4_PW_all += 1
    return conv3_4_PW_all
conv3_4_PW_all = 0
def cut_conv3_4_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv3_4_PW_zero()
            cut_conv3_4_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv3_5_DW_zero():
    global conv3_5_DW_zero
    conv3_5_DW_zero += 1
    return conv3_5_DW_zero
conv3_5_DW_zero = 0
def cut_conv3_5_DW_all():
    global conv3_5_DW_all
    conv3_5_DW_all += 1
    return conv3_5_DW_all
conv3_5_DW_all = 0
def cut_conv3_5_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv3_5_DW_zero()
            cut_conv3_5_DW_all()
    return x
def cut_conv3_5_PW_zero():
    global conv3_5_PW_zero
    conv3_5_PW_zero += 1
    return conv3_5_PW_zero
conv3_5_PW_zero = 0
def cut_conv3_5_PW_all():
    global conv3_5_PW_all
    conv3_5_PW_all += 1
    return conv3_5_PW_all
conv3_5_PW_all = 0
def cut_conv3_5_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv3_5_PW_zero()
            cut_conv3_5_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv4_1_DW_zero():
    global conv4_1_DW_zero
    conv4_1_DW_zero += 1
    return conv4_1_DW_zero
conv4_1_DW_zero = 0
def cut_conv4_1_DW_all():
    global conv4_1_DW_all
    conv4_1_DW_all += 1
    return conv4_1_DW_all
conv4_1_DW_all = 0
def cut_conv4_1_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            #是否全为0
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv4_1_DW_zero()
            cut_conv4_1_DW_all()
    # print('Zero num',zero_num_all,'All num',len(x[0]),'Zero percent:',100*float(zero_num_all) / len(x[0]),'%')
    return x
def cut_conv4_1_PW_zero():
    global conv4_1_PW_zero
    conv4_1_PW_zero += 1
    return conv4_1_PW_zero
conv4_1_PW_zero = 0
def cut_conv4_1_PW_all():
    global conv4_1_PW_all
    conv4_1_PW_all += 1
    return conv4_1_PW_all
conv4_1_PW_all = 0
def cut_conv4_1_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv4_1_PW_zero()
            cut_conv4_1_PW_all()
    return x
#-----------------------------------------------------------------------
def cut_conv4_2_DW_zero():
    global conv4_2_DW_zero
    conv4_2_DW_zero += 1
    return conv4_2_DW_zero
conv4_2_DW_zero = 0
def cut_conv4_2_DW_all():
    global conv4_2_DW_all
    conv4_2_DW_all += 1
    return conv4_2_DW_all
conv4_2_DW_all = 0
def cut_conv4_2_DW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<4)):
                x[m][k] = 0
                cut_conv4_2_DW_zero()
            cut_conv4_2_DW_all()
    return x
def cut_conv4_2_PW_zero():
    global conv4_2_PW_zero
    conv4_2_PW_zero += 1
    return conv4_2_PW_zero
conv4_2_PW_zero = 0
def cut_conv4_2_PW_all():
    global conv4_2_PW_all
    conv4_2_PW_all += 1
    return conv4_2_PW_all
conv4_2_PW_all = 0
def cut_conv4_2_PW(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(max_obs)<0.001)):
                x[m][k] = 0
                cut_conv4_2_PW_zero()
            cut_conv4_2_PW_all()
    return x
class ConvBNReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x)
        x = cut_conv1(x)
        return x

class DepthSeparableConv2d_conv2_1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv2_1, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv2_1_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv2_1_PW(x)
        return x
class DepthSeparableConv2d_conv2_2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv2_2, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv2_2_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv2_2_PW(x)
        return x
class DepthSeparableConv2d_conv2_3(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv2_3, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv2_3_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv2_3_PW(x)
        return x
class DepthSeparableConv2d_conv2_4(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv2_4, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv2_4_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv2_4_PW(x)
        return x
class DepthSeparableConv2d_conv2_5(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv2_5, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv2_5_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv2_5_PW(x)
        return x
    
    
class DepthSeparableConv2d_conv2_6(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv2_6, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv2_6_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv2_6_PW(x)
        return x

class DepthSeparableConv2d_conv3_1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv3_1, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv3_1_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv3_1_PW(x)
        return x
class DepthSeparableConv2d_conv3_2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv3_2, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv3_2_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv3_2_PW(x)
        return x
class DepthSeparableConv2d_conv3_3(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv3_3, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv3_3_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv3_3_PW(x)
        return x
class DepthSeparableConv2d_conv3_4(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv3_4, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv3_4_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv3_4_PW(x)
        return x
class DepthSeparableConv2d_conv3_5(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv3_5, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv3_5_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv3_5_PW(x)
        return x
class DepthSeparableConv2d_conv4_1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv4_1, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv4_1_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        x = cut_conv4_1_PW(x)
        return x
class DepthSeparableConv2d_conv4_2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(DepthSeparableConv2d_conv4_2, self).__init__()
        self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
        self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
        self.depthConv_ReLU = nn.ReLU(inplace=True)
        self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
        self.pointConv_ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthConv_Conv2d(x)
        x = self.depthConv_BatchNorm2d(x)
        x = self.depthConv_ReLU(x)
        x = cut_conv4_2_DW(x)
        x = self.pointConv_Conv2d(x)
        x = self.pointConv_BatchNorm2d(x)
        x = self.pointConv_ReLU(x)
        # print(x)
        x = cut_conv4_2_PW(x)
        return x

#深度可分离卷积块，由两部分组成，1.单核dw卷积，2.逐点卷积
# class DepthSeparableConv2d(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
#         super(DepthSeparableConv2d, self).__init__()
#         self.depthConv_Conv2d = nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, **kwargs)
#         self.depthConv_BatchNorm2d = nn.BatchNorm2d(in_channel)
#         self.depthConv_ReLU = nn.ReLU(inplace=True)
# 
#         self.pointConv_Conv2d = nn.Conv2d(in_channel, out_channel, (1, 1))
#         self.pointConv_BatchNorm2d = nn.BatchNorm2d(out_channel)
#         self.pointConv_ReLU = nn.ReLU(inplace=True)
# 
#     def forward(self, x):
#         x = self.depthConv_Conv2d(x)
#         x = self.depthConv_BatchNorm2d(x)
#         x = self.depthConv_ReLU(x)
#         # print('DW')
#         # x = quan_conv(x)
#         x = self.pointConv_Conv2d(x)
#         x = self.pointConv_BatchNorm2d(x)
#         x = self.pointConv_ReLU(x)
#         # print('PW')
#         # x = quan_conv(x)
#         return x

class MobileNet_v1(nn.Module):

    def __init__(self, in_channels=3, scale=1.0, num_classes=10, **kwargs):
        super(MobileNet_v1, self).__init__()
        input_channel = make_divisible(32)
        self.conv1 = ConvBNReLU(in_channels, input_channel, 3, 2, padding=1, bias=False)
        self.conv2_1 = DepthSeparableConv2d_conv2_1(32, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_2 = DepthSeparableConv2d_conv2_2(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv2_3 = DepthSeparableConv2d_conv2_3(128, 128, 3, stride=1, padding=1, bias=False)
        self.conv2_4 = DepthSeparableConv2d_conv2_4(128, 256, 3, stride=2, padding=1, bias=False)
        self.conv2_5 = DepthSeparableConv2d_conv2_5(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv2_6 = DepthSeparableConv2d_conv2_6(256, 512, 3, stride=2, padding=1, bias=False)
        input_channel = 512
        self.conv3_1 = DepthSeparableConv2d_conv3_1(input_channel, input_channel, 3, stride=1, padding=1, bias=False)
        self.conv3_2 = DepthSeparableConv2d_conv3_2(input_channel, input_channel, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = DepthSeparableConv2d_conv3_3(input_channel, input_channel, 3, stride=1, padding=1, bias=False)
        self.conv3_4 = DepthSeparableConv2d_conv3_4(input_channel, input_channel, 3, stride=1, padding=1, bias=False)
        self.conv3_5 = DepthSeparableConv2d_conv3_5(input_channel, input_channel, 3, stride=1, padding=1, bias=False)
        last_channel = make_divisible(1024)
        self.conv4_1 = DepthSeparableConv2d_conv4_1(input_channel, last_channel, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = DepthSeparableConv2d_conv4_2(last_channel, last_channel, 3, stride=2, padding=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_channel, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = quan_conv(x)
        print('conv1')
        x = self.conv1(x)
        print('conv1_zero_all:', conv1_zero, 'conv1_all:',conv1_all,'conv1_zero_percent:', conv1_zero / conv1_all)
        #conv2
        print('conv2_1:')
        x = self.conv2_1(x)
        print('conv2_1_dw_zero_all:', conv2_1_DW_zero, 'conv2_1_DW_all:', conv2_1_DW_all, 'conv2_1_DW_zero_percent:', conv2_1_DW_zero / conv2_1_DW_all)
        print('conv2_1_pw_zero_all:', conv2_1_PW_zero, 'conv2_1_PW_all:', conv2_1_PW_all, 'conv2_1_PW_zero_percent:', conv2_1_PW_zero / conv2_1_PW_all)
        print('conv2_2:')
        x = self.conv2_2(x)
        print('conv2_2_dw_zero_all:', conv2_2_DW_zero, 'conv2_2_DW_all:', conv2_2_DW_all, 'conv2_2_DW_zero_percent:', conv2_2_DW_zero / conv2_2_DW_all)
        print('conv2_2_pw_zero_all:', conv2_2_PW_zero, 'conv2_2_PW_all:', conv2_2_PW_all, 'conv2_2_PW_zero_percent:', conv2_2_PW_zero / conv2_2_PW_all)
        print('conv2_3')
        x = self.conv2_3(x)
        print('conv2_3_dw_zero_all:', conv2_3_DW_zero, 'conv2_3_DW_all:', conv2_3_DW_all, 'conv2_3_DW_zero_percent:',
              conv2_3_DW_zero / conv2_3_DW_all)
        print('conv2_3_pw_zero_all:', conv2_3_PW_zero, 'conv2_3_PW_all:', conv2_3_PW_all, 'conv2_3_PW_zero_percent:',
              conv2_3_PW_zero / conv2_3_PW_all)
        print('conv2_4')
        x = self.conv2_4(x)
        print('conv2_4_dw_zero_all:', conv2_4_DW_zero, 'conv2_4_DW_all:', conv2_4_DW_all, 'conv2_4_DW_zero_percent:',
              conv2_4_DW_zero / conv2_4_DW_all)
        print('conv2_4_pw_zero_all:', conv2_4_PW_zero, 'conv2_4_PW_all:', conv2_4_PW_all, 'conv2_4_PW_zero_percent:',
              conv2_4_PW_zero / conv2_4_PW_all)
        print('conv2_5')
        x = self.conv2_5(x)
        print('conv2_5_dw_zero_all:', conv2_5_DW_zero, 'conv2_5_DW_all:', conv2_5_DW_all, 'conv2_5_DW_zero_percent:',
              conv2_5_DW_zero / conv2_5_DW_all)
        print('conv2_5_pw_zero_all:', conv2_5_PW_zero, 'conv2_5_PW_all:', conv2_5_PW_all, 'conv2_5_PW_zero_percent:',
              conv2_5_PW_zero / conv2_5_PW_all)
        print('conv2_6')
        x = self.conv2_6(x)
        print('conv2_6_dw_zero_all:', conv2_6_DW_zero, 'conv2_6_DW_all:', conv2_6_DW_all, 'conv2_6_DW_zero_percent:',
              conv2_6_DW_zero / conv2_6_DW_all)
        print('conv2_6_pw_zero_all:', conv2_6_PW_zero, 'conv2_6_PW_all:', conv2_6_PW_all, 'conv2_6_PW_zero_percent:',
              conv2_6_PW_zero / conv2_6_PW_all)
        
        #conv3
        print('conv3_1')
        x = self.conv3_1(x)
        print('conv3_1_dw_zero_all:', conv3_1_DW_zero, 'conv3_1_DW_all:', conv3_1_DW_all, 'conv3_1_DW_zero_percent:',
              conv3_1_DW_zero / conv3_1_DW_all)
        print('conv3_1_pw_zero_all:', conv3_1_PW_zero, 'conv3_1_PW_all:', conv3_1_PW_all, 'conv3_1_PW_zero_percent:',
              conv3_1_PW_zero / conv3_1_PW_all)
        print('conv3_2')
        x = self.conv3_2(x)
        print('conv3_2_dw_zero_all:', conv3_2_DW_zero, 'conv3_2_DW_all:', conv3_2_DW_all, 'conv3_2_DW_zero_percent:',
              conv3_2_DW_zero / conv3_2_DW_all)
        print('conv3_2_pw_zero_all:', conv3_2_PW_zero, 'conv3_2_PW_all:', conv3_2_PW_all, 'conv3_2_PW_zero_percent:',
              conv3_2_PW_zero / conv3_2_PW_all)
        print('conv3_3')
        x = self.conv3_3(x)
        print('conv3_3_dw_zero_all:', conv3_3_DW_zero, 'conv3_3_DW_all:', conv3_3_DW_all, 'conv3_3_DW_zero_percent:',
              conv3_3_DW_zero / conv3_3_DW_all)
        print('conv3_3_pw_zero_all:', conv3_3_PW_zero, 'conv3_3_PW_all:', conv3_3_PW_all, 'conv3_3_PW_zero_percent:',
              conv3_3_PW_zero / conv3_3_PW_all)
        print('conv3_4')
        x = self.conv3_4(x)
        print('conv3_4_dw_zero_all:', conv3_4_DW_zero, 'conv3_4_DW_all:', conv3_4_DW_all, 'conv3_4_DW_zero_percent:',
              conv3_4_DW_zero / conv3_4_DW_all)
        print('conv3_4_pw_zero_all:', conv3_4_PW_zero, 'conv3_4_PW_all:', conv3_4_PW_all, 'conv3_4_PW_zero_percent:',
              conv3_4_PW_zero / conv3_4_PW_all)
        print('conv3_5')
        x = self.conv3_5(x)
        print('conv3_5_dw_zero_all:', conv3_5_DW_zero, 'conv3_5_DW_all:', conv3_5_DW_all, 'conv3_5_DW_zero_percent:',
              conv3_5_DW_zero / conv3_5_DW_all)
        print('conv3_5_pw_zero_all:', conv3_5_PW_zero, 'conv3_5_PW_all:', conv3_5_PW_all, 'conv3_5_PW_zero_percent:',
              conv3_5_PW_zero / conv3_5_PW_all)
        #conv4
        print('conv4_1')
        x = self.conv4_1(x)
        print('conv4_1_dw_zero_all:', conv4_1_DW_zero, 'conv4_1_DW_all:', conv4_1_DW_all, 'conv4_1_DW_zero_percent:',
              conv4_1_DW_zero / conv4_1_DW_all)
        print('conv4_1_pw_zero_all:', conv4_1_PW_zero, 'conv4_1_PW_all:', conv4_1_PW_all, 'conv4_1_PW_zero_percent:',
              conv4_1_PW_zero / conv4_1_PW_all)
        print('conv4_2')
        x = self.conv4_2(x)
        # print(x)
        print('conv4_2_dw_zero_all:', conv4_2_DW_zero, 'conv4_2_DW_all:', conv4_2_DW_all, 'conv4_2_DW_zero_percent:',
              conv4_2_DW_zero / conv4_2_DW_all)
        print('conv4_2_pw_zero_all:', conv4_2_PW_zero, 'conv4_2_PW_all:', conv4_2_PW_all, 'conv4_2_PW_zero_percent:',
              conv4_2_PW_zero / conv4_2_PW_all)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
