import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torchsummary import summary
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
            # if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20))://稀疏剪枝
            # if (torch.eq(max_obs, 0) == 1)://全零剪枝
            # if (max_obs <= 1)://最值剪枝，后续各网络层剪枝条件同理
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv1_zero()
            cut_conv1_all()
    return x
#-----------------------------------------------------------------------
def cut_conv2_zero():
    global conv2_zero
    conv2_zero += 1
    return conv2_zero
conv2_zero = 0
def cut_conv2_all():
    global conv2_all
    conv2_all += 1
    return conv2_all
conv2_all = 0
def cut_conv2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv2_zero()
            cut_conv2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv3_zero():
    global conv3_zero
    conv3_zero += 1
    return conv3_zero
conv3_zero = 0
def cut_conv3_all():
    global conv3_all
    conv3_all += 1
    return conv3_all
conv3_all = 0
def cut_conv3(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv3_zero()
            cut_conv3_all()
    return x
#-----------------------------------------------------------------------
def cut_conv4_zero():
    global conv4_zero
    conv4_zero += 1
    return conv4_zero
conv4_zero = 0
def cut_conv4_all():
    global conv4_all
    conv4_all += 1
    return conv4_all
conv4_all = 0
def cut_conv4(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv4_zero()
            cut_conv4_all()
    return x
#-----------------------------------------------------------------------
def cut_conv5_zero():
    global conv5_zero
    conv5_zero += 1
    return conv5_zero
conv5_zero = 0
def cut_conv5_all():
    global conv5_all
    conv5_all += 1
    return conv5_all
conv5_all = 0
def cut_conv5(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv5_zero()
            cut_conv5_all()
    return x
#-----------------------------------------------------------------------
def cut_conv6_zero():
    global conv6_zero
    conv6_zero += 1
    return conv6_zero
conv6_zero = 0
def cut_conv6_all():
    global conv6_all
    conv6_all += 1
    return conv6_all
conv6_all = 0
def cut_conv6(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv6_zero()
            cut_conv6_all()
    return x
#-----------------------------------------------------------------------
def cut_conv7_zero():
    global conv7_zero
    conv7_zero += 1
    return conv7_zero
conv7_zero = 0
def cut_conv7_all():
    global conv7_all
    conv7_all += 1
    return conv7_all
conv7_all = 0
def cut_conv7(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv7_zero()
            cut_conv7_all()
    return x
#-----------------------------------------------------------------------
def cut_conv8_zero():
    global conv8_zero
    conv8_zero += 1
    return conv8_zero
conv8_zero = 0
def cut_conv8_all():
    global conv8_all
    conv8_all += 1
    return conv8_all
conv8_all = 0
def cut_conv8(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv8_zero()
            cut_conv8_all()
    return x
#-----------------------------------------------------------------------
def cut_conv9_zero():
    global conv9_zero
    conv9_zero += 1
    return conv9_zero
conv9_zero = 0
def cut_conv9_all():
    global conv9_all
    conv9_all += 1
    return conv9_all
conv9_all = 0
def cut_conv9(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv9_zero()
            cut_conv9_all()
    return x

#-----------------------------------------------------------------------
def cut_conv2_2_zero():
    global conv2_2_zero
    conv2_2_zero += 1
    return conv2_2_zero
conv2_2_zero = 0
def cut_conv2_2_all():
    global conv2_2_all
    conv2_2_all += 1
    return conv2_2_all
conv2_2_all = 0
def cut_conv2_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv2_2_zero()
            cut_conv2_2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv3_2_zero():
    global conv3_2_zero
    conv3_2_zero += 1
    return conv3_2_zero
conv3_2_zero = 0
def cut_conv3_2_all():
    global conv3_2_all
    conv3_2_all += 1
    return conv3_2_all
conv3_2_all = 0
def cut_conv3_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv3_2_zero()
            cut_conv3_2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv4_2_zero():
    global conv4_2_zero
    conv4_2_zero += 1
    return conv4_2_zero
conv4_2_zero = 0
def cut_conv4_2_all():
    global conv4_2_all
    conv4_2_all += 1
    return conv4_2_all
conv4_2_all = 0
def cut_conv4_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv4_2_zero()
            cut_conv4_2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv5_2_zero():
    global conv5_2_zero
    conv5_2_zero += 1
    return conv5_2_zero
conv5_2_zero = 0
def cut_conv5_2_all():
    global conv5_2_all
    conv5_2_all += 1
    return conv5_2_all
conv5_2_all = 0
def cut_conv5_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv5_2_zero()
            cut_conv5_2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv6_2_zero():
    global conv6_2_zero
    conv6_2_zero += 1
    return conv6_2_zero
conv6_2_zero = 0
def cut_conv6_2_all():
    global conv6_2_all
    conv6_2_all += 1
    return conv6_2_all
conv6_2_all = 0
def cut_conv6_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv6_2_zero()
            cut_conv6_2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv7_2_zero():
    global conv7_2_zero
    conv7_2_zero += 1
    return conv7_2_zero
conv7_2_zero = 0
def cut_conv7_2_all():
    global conv7_2_all
    conv7_2_all += 1
    return conv7_2_all
conv7_2_all = 0
def cut_conv7_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv7_2_zero()
            cut_conv7_2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv8_2_zero():
    global conv8_2_zero
    conv8_2_zero += 1
    return conv8_2_zero
conv8_2_zero = 0
def cut_conv8_2_all():
    global conv8_2_all
    conv8_2_all += 1
    return conv8_2_all
conv8_2_all = 0
def cut_conv8_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv8_2_zero()
            cut_conv8_2_all()
    return x
#-----------------------------------------------------------------------
def cut_conv9_2_zero():
    global conv9_2_zero
    conv9_2_zero += 1
    return conv9_2_zero
conv9_2_zero = 0
def cut_conv9_2_all():
    global conv9_2_all
    conv9_2_all += 1
    return conv9_2_all
conv9_2_all = 0
def cut_conv9_2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_conv9_2_zero()
            cut_conv9_2_all()
    return x
#-----------------------------------------------------------------------
def cut_en0_zero():
    global en0_zero
    en0_zero += 1
    return en0_zero
en0_zero = 0
def cut_en0_all():
    global en0_all
    en0_all += 1
    return en0_all
en0_all = 0
def cut_en0(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_en0_zero()
            cut_en0_all()
    return x
#-----------------------------------------------------------------------
def cut_en1_zero():
    global en1_zero
    en1_zero += 1
    return en1_zero
en1_zero = 0
def cut_en1_all():
    global en1_all
    en1_all += 1
    return en1_all
en1_all = 0
def cut_en1(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_en1_zero()
            cut_en1_all()
    return x
#-----------------------------------------------------------------------
def cut_en2_zero():
    global en2_zero
    en2_zero += 1
    return en2_zero
en2_zero = 0
def cut_en2_all():
    global en2_all
    en2_all += 1
    return en2_all
en2_all = 0
def cut_en2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_en2_zero()
            cut_en2_all()
    return x
#-----------------------------------------------------------------------
def cut_en3_zero():
    global en3_zero
    en3_zero += 1
    return en3_zero
en3_zero = 0
def cut_en3_all():
    global en3_all
    en3_all += 1
    return en3_all
en3_all = 0
def cut_en3(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_en3_zero()
            cut_en3_all()
    return x
#-----------------------------------------------------------------------
def cut_cut_resdual1_zero():
    global cut_resdual1_zero
    cut_resdual1_zero += 1
    return cut_resdual1_zero
cut_resdual1_zero = 0
def cut_cut_resdual1_all():
    global cut_resdual1_all
    cut_resdual1_all += 1
    return cut_resdual1_all
cut_resdual1_all = 0
def cut_resdual1(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_cut_resdual1_zero()
            cut_cut_resdual1_all()
    return x
#-----------------------------------------------------------------------
def cut_cut_resdual2_zero():
    global cut_resdual2_zero
    cut_resdual2_zero += 1
    return cut_resdual2_zero
cut_resdual2_zero = 0
def cut_cut_resdual2_all():
    global cut_resdual2_all
    cut_resdual2_all += 1
    return cut_resdual2_all
cut_resdual2_all = 0
def cut_resdual2(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_cut_resdual2_zero()
            cut_cut_resdual2_all()
    return x
#-----------------------------------------------------------------------
def cut_cut_resdual3_zero():
    global cut_resdual3_zero
    cut_resdual3_zero += 1
    return cut_resdual3_zero
cut_resdual3_zero = 0
def cut_cut_resdual3_all():
    global cut_resdual3_all
    cut_resdual3_all += 1
    return cut_resdual3_all
cut_resdual3_all = 0
def cut_resdual3(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_cut_resdual3_zero()
            cut_cut_resdual3_all()
    return x
#-----------------------------------------------------------------------
def cut_cut_resdual4_zero():
    global cut_resdual4_zero
    cut_resdual4_zero += 1
    return cut_resdual4_zero
cut_resdual4_zero = 0
def cut_cut_resdual4_all():
    global cut_resdual4_all
    cut_resdual4_all += 1
    return cut_resdual4_all
cut_resdual4_all = 0
def cut_resdual4(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_cut_resdual4_zero()
            cut_cut_resdual4_all()
    return x

#-----------------------------------------------------------------------
def cut_cut_resdual5_zero():
    global cut_resdual5_zero
    cut_resdual5_zero += 1
    return cut_resdual5_zero
cut_resdual5_zero = 0
def cut_cut_resdual5_all():
    global cut_resdual5_all
    cut_resdual5_all += 1
    return cut_resdual5_all
cut_resdual5_all = 0
def cut_resdual5(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_cut_resdual5_zero()
            cut_cut_resdual5_all()
    return x
#-----------------------------------------------------------------------
def cut_cut_resdual6_zero():
    global cut_resdual6_zero
    cut_resdual6_zero += 1
    return cut_resdual6_zero
cut_resdual6_zero = 0
def cut_cut_resdual6_all():
    global cut_resdual6_all
    cut_resdual6_all += 1
    return cut_resdual6_all
cut_resdual6_all = 0
def cut_resdual6(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_cut_resdual6_zero()
            cut_cut_resdual6_all()
    return x
#-----------------------------------------------------------------------
def cut_cut_resdual7_zero():
    global cut_resdual7_zero
    cut_resdual7_zero += 1
    return cut_resdual7_zero
cut_resdual7_zero = 0
def cut_cut_resdual7_all():
    global cut_resdual7_all
    cut_resdual7_all += 1
    return cut_resdual7_all
cut_resdual7_all = 0
def cut_resdual7(x):
    for m in range(len(x)):
        for k in range(len(x[0])):
            max = torch.max(x[m][k])
            min = torch.min(x[m][k])
            max_obs = torch.max(max, (-min))
            if ((float(torch.count_nonzero(x[m][k]).item()) / (len(x[0][0]) * len(x[0][0][0])) < 0.20)):
                x[m][k] = 0
                cut_cut_resdual7_zero()
            cut_cut_resdual7_all()
    return x

class Resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet18, self).__init__()
        self.model0_Conv2d = Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3)
        self.model0_BatchNorm2d = BatchNorm2d(64)
        self.model0_ReLU = ReLU()
        self.model0_MaxPool2d = MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.model1_Conv2d = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.model1_BatchNorm2d = BatchNorm2d(64)
        self.model1_ReLU = ReLU()
        self.model1_2Conv2d = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.model1_2BatchNorm2d = BatchNorm2d(64)
        self.model1_2ReLU = ReLU()

        self.R1 = ReLU()


        self.model2_Conv2d = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.model2_BatchNorm2d = BatchNorm2d(64)
        self.model2_ReLU = ReLU()
        self.model2_2Conv2d = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.model2_2BatchNorm2d = BatchNorm2d(64)
        self.model2_2ReLU = ReLU()
        self.R2 = ReLU()


        self.model3_Conv2d = Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        self.model3_BatchNorm2d = BatchNorm2d(128)
        self.model3_ReLU = ReLU()
        self.model3_2Conv2d = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.model3_2BatchNorm2d = BatchNorm2d(128)
        self.model3_2ReLU = ReLU()

        self.en1_Conv2d = Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0)
        self.en1_BatchNorm2d = BatchNorm2d(128)
        self.en1_ReLU = ReLU()

        self.R3 = ReLU()

        self.model4_Conv2d = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.model4_BatchNorm2d = BatchNorm2d(128)
        self.model4_ReLU = ReLU()
        self.model4_2Conv2d = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.model4_2BatchNorm2d = BatchNorm2d(128)
        self.model4_2ReLU = ReLU()

        self.R4 = ReLU()

        self.model5_Conv2d = Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1)
        self.model5_BatchNorm2d = BatchNorm2d(256)
        self.model5_ReLU = ReLU()
        self.model5_2Conv2d = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.model5_2BatchNorm2d = BatchNorm2d(256)
        self.model5_2ReLU = ReLU()

        self.en2_Conv2d = Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0)
        self.en2_BatchNorm2d = BatchNorm2d(256)
        self.en2_ReLU = ReLU()

        self.R5 = ReLU()

        self.model6_Conv2d = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.model6_BatchNorm2d = BatchNorm2d(256)
        self.model6_ReLU = ReLU()
        self.model6_2Conv2d = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.model6_2BatchNorm2d = BatchNorm2d(256)
        self.model6_2ReLU = ReLU()

        self.R6 = ReLU()

        self.model7_Conv2d = Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1)
        self.model7_BatchNorm2d = BatchNorm2d(512)
        self.model7_ReLU = ReLU()
        self.model7_2Conv2d = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.model7_2BatchNorm2d = BatchNorm2d(512)
        self.model7_2ReLU = ReLU()

        self.en3_Conv2d = Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0)
        self.en3_BatchNorm2d = BatchNorm2d(512)
        self.en3_ReLU = ReLU()

        self.R7 = ReLU()

        self.model8_Conv2d = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.model8_BatchNorm2d = BatchNorm2d(512)
        self.model8_ReLU = ReLU()
        self.model8_2Conv2d = Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.model8_2BatchNorm2d = BatchNorm2d(512)
        self.model8_2ReLU = ReLU()

        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        #print('conv1:')
        x = self.model0_Conv2d(x)
        x = self.model0_BatchNorm2d(x)
        x = self.model0_ReLU(x)

        x = cut_conv1(x)
        print('conv1_zero_all:', conv1_zero, 'conv1_all:', conv1_all, 'conv1_zero_percent:', conv1_zero / conv1_all)

        x = self.model0_MaxPool2d(x)

        f1 = x
        # x = self.model1(x)
        #print('conv2:')
        x = self.model1_Conv2d(x)
        x = self.model1_BatchNorm2d(x)
        x = self.model1_ReLU(x)

        x = cut_conv2(x)
        print('conv2_1_zero_all:', conv2_zero, 'conv2_1_all:', conv2_all, 'conv2_1_zero_percent:', conv2_zero / conv2_all)

        x = self.model1_2Conv2d(x)
        x = self.model1_2BatchNorm2d(x)
        x = self.model1_2ReLU(x)

        x = cut_conv2_2(x)
        print('conv2_2_zero_all:', conv2_2_zero, 'conv2_2_all:', conv2_2_all, 'conv2_2_zero_percent:', conv2_2_zero / conv2_2_all)

        x = x + f1
        x = self.R1(x)



        f1_1 = cut_en0(x)
        print('f1_1_all:', en0_zero, 'f1_1_all:', en0_all, 'f1_1_zero_percent:', en0_zero / en0_all)

        # x = self.model2(x)
        #print('conv3:')
        x = self.model2_Conv2d(x)
        x = self.model2_BatchNorm2d(x)
        x = self.model2_ReLU(x)

        x = cut_conv3(x)
        print('conv3_1_zero_all:', conv3_zero, 'conv3_1_all:', conv3_all, 'conv3_1_zero_percent:', conv3_zero / conv3_all)

        x = self.model2_2Conv2d(x)
        x = self.model2_2BatchNorm2d(x)
        x = self.model2_2ReLU(x)

        x = cut_conv3_2(x)
        print('conv3_2_zero_all:', conv3_2_zero, 'conv3_2_all:', conv3_2_all, 'conv3_2_zero_percent:', conv3_2_zero / conv3_2_all)

        x = x + f1_1
        x = self.R2(x)

        x = cut_resdual1(x)
        print('cut_resdual1_all:', cut_resdual1_zero, 'cut_resdual1_all:', cut_resdual1_all, 'cut_resdual1_zero_percent:', cut_resdual1_zero / cut_resdual1_all)

        # f2_1 = x
        # f2_1 = self.en1(f2_1)
        #print('en1:')
        f2_1 = self.en1_Conv2d(x)
        f2_1 = self.en1_BatchNorm2d(f2_1)
        f2_1 = self.en1_ReLU(f2_1)

        f2_1 = cut_en1(f2_1)
        print('f2_1_all:', en1_zero, 'f2_1_all:', en1_all, 'f2_1_zero_percent:', en1_zero / en1_all)

        # x = self.model3(x)
        #print('conv4:')
        x = self.model3_Conv2d(x)
        x = self.model3_BatchNorm2d(x)
        x = self.model3_ReLU(x)

        x = cut_conv4(x)
        print('conv4_1_zero_all:', conv4_zero, 'conv4_1_all:', conv4_all, 'conv4_1_zero_percent:', conv4_zero / conv4_all)

        x = self.model3_2Conv2d(x)
        x = self.model3_2BatchNorm2d(x)
        x = self.model3_2ReLU(x)

        x = cut_conv4_2(x)
        print('conv4_2_zero_all:', conv4_2_zero, 'conv4_2_all:', conv4_2_all, 'conv4_2_zero_percent:', conv4_2_zero / conv4_2_all)

        x = x + f2_1
        x = self.R3(x)

        x = cut_resdual2(x)
        print('cut_resdual2_all:', cut_resdual2_zero, 'cut_resdual2_all:', cut_resdual2_all, 'cut_resdual2_zero_percent:', cut_resdual2_zero / cut_resdual2_all)


        f2_2 = x


        # x = self.model4(x)
        #print('conv5:')
        x = self.model4_Conv2d(x)
        x = self.model4_BatchNorm2d(x)
        x = self.model4_ReLU(x)

        x = cut_conv5(x)
        print('conv5_1_zero_all:', conv5_zero, 'conv5_1_all:', conv5_all, 'conv5_1_zero_percent:', conv5_zero / conv5_all)

        x = self.model4_2Conv2d(x)
        x = self.model4_2BatchNorm2d(x)
        x = self.model4_2ReLU(x)

        x = cut_conv5_2(x)
        print('conv5_2_zero_all:', conv5_2_zero, 'conv5_2_all:', conv5_2_all, 'conv5_2_zero_percent:', conv5_2_zero / conv5_2_all)

        x = x + f2_2
        x = self.R4(x)
        x = cut_resdual3(x)
        print('cut_resdual3_all:', cut_resdual3_zero, 'cut_resdual3_all:', cut_resdual3_all, 'cut_resdual3_zero_percent:', cut_resdual3_zero / cut_resdual3_all)

        # f3_1 = x
        # f3_1 = self.en2(f3_1)
        #print('en2:')
        f3_1 = self.en2_Conv2d(x)
        f3_1 = self.en2_BatchNorm2d(f3_1)
        f3_1 = self.en2_ReLU(f3_1)

        f3_1 = cut_en2(f3_1)
        print('f3_1_all:', en2_zero, 'f3_1_all:', en2_all, 'f3_1_zero_percent:', en2_zero / en2_all)

        # x = self.model5(x)
        #print('conv6:')
        x = self.model5_Conv2d(x)
        x = self.model5_BatchNorm2d(x)
        x = self.model5_ReLU(x)

        x = cut_conv6(x)
        print('conv6_1_zero_all:', conv6_zero, 'conv6_1_all:', conv6_all, 'conv6_1_zero_percent:', conv6_zero / conv6_all)

        x = self.model5_2Conv2d(x)
        x = self.model5_2BatchNorm2d(x)
        x = self.model5_2ReLU(x)

        x = cut_conv6_2(x)
        print('conv6_2_zero_all:', conv6_2_zero, 'conv6_2_all:', conv6_2_all, 'conv6_2_zero_percent:', conv6_2_zero / conv6_2_all)

        x = x + f3_1
        x = self.R5(x)
        x = cut_resdual4(x)
        print('cut_resdual4_all:', cut_resdual4_zero, 'cut_resdual4_all:', cut_resdual4_all, 'cut_resdual4_zero_percent:', cut_resdual4_zero / cut_resdual4_all)

        f3_2 = x
        # x = self.model6(x)
        #print('conv7:')
        x = self.model6_Conv2d(x)
        x = self.model6_BatchNorm2d(x)
        x = self.model6_ReLU(x)

        x = cut_conv7(x)
        print('conv7_1_zero_all:', conv7_zero, 'conv7_1_all:', conv7_all, 'conv7_1_zero_percent:', conv7_zero / conv7_all)

        x = self.model6_2Conv2d(x)
        x = self.model6_2BatchNorm2d(x)
        x = self.model6_2ReLU(x)

        x = cut_conv7_2(x)
        print('conv7_2_zero_all:', conv7_2_zero, 'conv7_2_all:', conv7_2_all, 'conv7_2_zero_percent:', conv7_2_zero / conv7_2_all)

        x = x + f3_2
        x = self.R6(x)
        x = cut_resdual5(x)
        print('cut_resdual5_all:', cut_resdual5_zero, 'cut_resdual5_all:', cut_resdual5_all, 'cut_resdual5_zero_percent:', cut_resdual5_zero / cut_resdual5_all)

        # f4_1 = x
        # f4_1 = self.en3(f4_1)
        f4_1 = self.en3_Conv2d(x)
        f4_1 = self.en3_BatchNorm2d(f4_1)
        f4_1 = self.en3_ReLU(f4_1)

        f4_1 = cut_en3(f4_1)
        print('f4_1_all:', en3_zero, 'f4_1_all:', en3_all, 'f4_1_zero_percent:', en3_zero / en3_all)

        # x = self.model7(x)
        #print('conv8:')
        x = self.model7_Conv2d(x)
        x = self.model7_BatchNorm2d(x)
        x = self.model7_ReLU(x)

        x = cut_conv8(x)
        print('conv8_1_zero_all:', conv8_zero, 'conv8_1_all:', conv8_all, 'conv8_1_zero_percent:', conv8_zero / conv8_all)

        x = self.model7_2Conv2d(x)
        x = self.model7_2BatchNorm2d(x)
        x = self.model7_2ReLU(x)

        x = cut_conv8_2(x)
        print('conv8_2_zero_all:', conv8_2_zero, 'conv8_2_all:', conv8_2_all, 'conv8_2_zero_percent:', conv8_2_zero / conv8_2_all)

        x = x + f4_1
        x = self.R7(x)
        x = cut_resdual6(x)
        print('cut_resdual6_all:', cut_resdual6_zero, 'cut_resdual6_all:', cut_resdual6_all, 'cut_resdual6_zero_percent:', cut_resdual6_zero / cut_resdual6_all)

        f4_2 = x
        # x = self.model8(x)
        # print('conv9:')
        x = self.model8_Conv2d(x)
        x = self.model8_BatchNorm2d(x)
        x = self.model8_ReLU(x)

        x = cut_conv9(x)
        print('conv9_1_zero_all:', conv9_zero, 'conv9_1_all:', conv9_all, 'conv9_1_zero_percent:', conv9_zero / conv9_all)

        x = self.model8_2Conv2d(x)
        x = self.model8_2BatchNorm2d(x)
        x = self.model8_2ReLU(x)

        x = cut_conv9_2(x)
        print('conv9_2_zero_all:', conv9_2_zero, 'conv9_2_all:', conv9_2_all, 'conv9_2_zero_percent:', conv9_2_zero / conv9_2_all)

        x = x + f4_2
        x = self.R8(x)
        x = cut_resdual7(x)
        print('cut_resdual7_all:', cut_resdual7_zero, 'cut_resdual7_all:', cut_resdual7_all, 'fcut_resdual7_zero_percent:', cut_resdual7_zero / cut_resdual7_all)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)

        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 10分类
    res18 = Resnet18().to('cuda:0')
    summary(res18, (3, 224, 224))
