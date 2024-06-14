import torch
from PIL import Image
import torchvision.transforms as transforms
from net import MobileNet_v2
# from cifra_nano_net import tinymcunet
from resnet18yzp import resnet18
from torchvision.models.resnet import resnet50
from torchvision.models.mobilenetv2 import mobilenet_v2
import argparse
import time
import pytest
from torch import nn
import os

def quantify_ten(input_data,min_ci_fang):
    input_data_shift = input_data << (19 - min_ci_fang)
    output = torch.round(input_data_shift)>>(19 - min_ci_fang)
    return output

def quantify(input_data,min_ci_fang):
    output = 0
    # print("Range:", pow(2, min_ci_fang - 12), "Min_unit:", pow(2, min_ci_fang - 19))
    # print(input_data)
    if (input_data < 0):
        input_data = -input_data
        for i in range(8):
            if (input_data >= pow(2, min_ci_fang - 12 - i)):
                input_data = input_data - pow(2, min_ci_fang - 12 - i)
                output = output -  pow(2, min_ci_fang - 12 - i)
    else:
        for i in range(8):
            if (input_data >= pow(2, min_ci_fang - 12 - i)):
                input_data = input_data - pow(2, min_ci_fang - 12 - i)
                output = output + pow(2, min_ci_fang - 12 - i)
    # print(output)
    return output

if __name__ == '__main__':
    quan_1d = [

        'features.0.1.weight',
        'features.0.1.bias',

        'features.1.conv.0.1.weight',
        'features.1.conv.0.1.bias',

        'features.1.conv.2.weight',
        'features.1.conv.2.bias',

        'features.2.conv.0.1.weight',
        'features.2.conv.0.1.bias',

        'features.2.conv.1.1.weight',
        'features.2.conv.1.1.bias',

        'features.2.conv.3.weight',
        'features.2.conv.3.bias',

        'features.3.conv.0.1.weight',
        'features.3.conv.0.1.bias',

        'features.3.conv.1.1.weight',
        'features.3.conv.1.1.bias',

        'features.3.conv.3.weight',
        'features.3.conv.3.bias',

        'features.4.conv.0.1.weight',
        'features.4.conv.0.1.bias',

        'features.4.conv.1.1.weight',
        'features.4.conv.1.1.bias',

        'features.4.conv.3.weight',
        'features.4.conv.3.bias',

        'features.5.conv.0.1.weight',
        'features.5.conv.0.1.bias',

        'features.5.conv.1.1.weight',
        'features.5.conv.1.1.bias',

        'features.5.conv.3.weight',
        'features.5.conv.3.bias',

        'features.6.conv.0.1.weight',
        'features.6.conv.0.1.bias',

        'features.6.conv.1.1.weight',
        'features.6.conv.1.1.bias',

        'features.6.conv.3.weight',
        'features.6.conv.3.bias',

        'features.7.conv.0.1.weight',
        'features.7.conv.0.1.bias',

        'features.7.conv.1.1.weight',
        'features.7.conv.1.1.bias',

        'features.7.conv.3.weight',
        'features.7.conv.3.bias',

        'features.8.conv.0.1.weight',
        'features.8.conv.0.1.bias',

        'features.8.conv.1.1.weight',
        'features.8.conv.1.1.bias',

        'features.8.conv.3.weight',
        'features.8.conv.3.bias',

        'features.9.conv.0.1.weight',
        'features.9.conv.0.1.bias',

        'features.9.conv.1.1.weight',
        'features.9.conv.1.1.bias',

        'features.9.conv.3.weight',
        'features.9.conv.3.bias',

        'features.10.conv.0.1.weight',
        'features.10.conv.0.1.bias',

        'features.10.conv.1.1.weight',
        'features.10.conv.1.1.bias',

        'features.10.conv.3.weight',
        'features.10.conv.3.bias',

        'features.11.conv.0.1.weight',
        'features.11.conv.0.1.bias',

        'features.11.conv.1.1.weight',
        'features.11.conv.1.1.bias',

        'features.11.conv.3.weight',
        'features.11.conv.3.bias',

        'features.12.conv.0.1.weight',
        'features.12.conv.0.1.bias',

        'features.12.conv.1.1.weight',
        'features.12.conv.1.1.bias',

        'features.12.conv.3.weight',
        'features.12.conv.3.bias',

        'features.13.conv.0.1.weight',
        'features.13.conv.0.1.bias',

        'features.13.conv.1.1.weight',
        'features.13.conv.1.1.bias',

        'features.13.conv.3.weight',
        'features.13.conv.3.bias',

        'features.14.conv.0.1.weight',
        'features.14.conv.0.1.bias',

        'features.14.conv.1.1.weight',
        'features.14.conv.1.1.bias',

        'features.14.conv.3.weight',
        'features.14.conv.3.bias',

        'features.15.conv.0.1.weight',
        'features.15.conv.0.1.bias',

        'features.15.conv.1.1.weight',
        'features.15.conv.1.1.bias',

        'features.15.conv.3.weight',
        'features.15.conv.3.bias',

        'features.16.conv.0.1.weight',
        'features.16.conv.0.1.bias',

        'features.16.conv.1.1.weight',
        'features.16.conv.1.1.bias',

        'features.16.conv.3.weight',
        'features.16.conv.3.bias',

        'features.17.conv.0.1.weight',
        'features.17.conv.0.1.bias',

        'features.17.conv.1.1.weight',
        'features.17.conv.1.1.bias',

        'features.17.conv.3.weight',
        'features.17.conv.3.bias',

        'features.18.1.weight',
        'features.18.1.bias',

        'classifier.1.bias'
        # 'bn1.weight',
        # 'bn1.bias',
        # 'layer1.0.bn1.weight',
        # 'layer1.0.bn1.bias',
        # 'layer1.0.bn2.weight',
        # 'layer1.0.bn2.bias',
        # 'layer1.0.bn3.weight',
        # 'layer1.0.bn3.bias',
        # 'layer1.1.bn1.weight',
        # 'layer1.1.bn1.bias',
        # 'layer1.1.bn2.weight',
        # 'layer1.1.bn2.bias',
        # 'layer1.1.bn3.weight',
        # 'layer1.1.bn3.bias',
        # 'layer1.2.bn1.weight',
        # 'layer1.2.bn1.bias',
        # 'layer1.2.bn2.weight',
        # 'layer1.2.bn2.bias',
        # 'layer1.2.bn3.weight',
        # 'layer1.2.bn3.bias',
        # 'layer2.0.bn1.weight',
        # 'layer2.0.bn1.bias',
        # 'layer2.0.bn2.weight',
        # 'layer2.0.bn2.bias',
        # 'layer2.0.bn3.weight',
        # 'layer2.0.bn3.bias',
        # 'layer2.0.downsample.1.weight',
        # 'layer2.0.downsample.1.bias',
        # 'layer2.1.bn1.weight',
        # 'layer2.1.bn1.bias',
        # 'layer2.1.bn2.weight',
        # 'layer2.1.bn2.bias',
        # 'layer2.1.bn3.weight',
        # 'layer2.1.bn3.bias',
        # 'layer2.2.bn1.weight',
        # 'layer2.2.bn1.bias',
        # 'layer2.2.bn2.weight',
        # 'layer2.2.bn2.bias',
        # 'layer2.2.bn3.weight',
        # 'layer2.2.bn3.bias',
        # 'layer2.3.bn1.weight',
        # 'layer2.3.bn1.bias',
        # 'layer2.3.bn2.weight',
        # 'layer2.3.bn2.bias',
        # 'layer2.3.bn3.weight',
        # 'layer2.3.bn3.bias',
        # 'layer3.0.bn1.weight',
        # 'layer3.0.bn1.bias',
        # 'layer3.0.bn2.weight',
        # 'layer3.0.bn2.bias',
        # 'layer3.0.bn3.weight',
        # 'layer3.0.bn3.bias',
        # 'layer3.0.downsample.1.weight',
        # 'layer3.0.downsample.1.bias',
        # 'layer3.1.bn1.weight',
        # 'layer3.1.bn1.bias',
        # 'layer3.1.bn2.weight',
        # 'layer3.1.bn2.bias',
        # 'layer3.1.bn3.weight',
        # 'layer3.1.bn3.bias',
        # 'layer3.2.bn1.weight',
        # 'layer3.2.bn1.bias',
        # 'layer3.2.bn2.weight',
        # 'layer3.2.bn2.bias',
        # 'layer3.2.bn3.weight',
        # 'layer3.2.bn3.bias',
        # 'layer3.3.bn1.weight',
        # 'layer3.3.bn1.bias',
        # 'layer3.3.bn2.weight',
        # 'layer3.3.bn2.bias',
        # 'layer3.3.bn3.weight',
        # 'layer3.3.bn3.bias',
        # 'layer3.4.bn1.weight',
        # 'layer3.4.bn1.bias',
        # 'layer3.4.bn2.weight',
        # 'layer3.4.bn2.bias',
        # 'layer3.4.bn3.weight',
        # 'layer3.4.bn3.bias',
        # 'layer3.5.bn1.weight',
        # 'layer3.5.bn1.bias',
        # 'layer3.5.bn2.weight',
        # 'layer3.5.bn2.bias',
        # 'layer3.5.bn3.weight',
        # 'layer3.5.bn3.bias',
        # 'layer4.0.bn1.weight',
        # 'layer4.0.bn1.bias',
        # 'layer4.0.bn2.weight',
        # 'layer4.0.bn2.bias',
        # 'layer4.0.bn3.weight',
        # 'layer4.0.bn3.bias',
        # 'layer4.0.downsample.1.weight',
        # 'layer4.0.downsample.1.bias',
        # 'layer4.1.bn1.weight',
        # 'layer4.1.bn1.bias',
        # 'layer4.1.bn2.weight',
        # 'layer4.1.bn2.bias',
        # 'layer4.1.bn3.weight',
        # 'layer4.1.bn3.bias',
        # 'layer4.2.bn1.weight',
        # 'layer4.2.bn1.bias',
        # 'layer4.2.bn2.weight',
        # 'layer4.2.bn2.bias',
        # 'layer4.2.bn3.weight',
        # 'layer4.2.bn3.bias',
        # 'fc.bias',
    ]
    quan_pw1d =[


    ]
    quan_2d = [
        'features.0.0.weight',
        'features.1.conv.0.0.weight',

        'features.1.conv.1.weight',

        'features.2.conv.0.0.weight',

        'features.2.conv.1.0.weight',

        'features.2.conv.2.weight',

        'features.3.conv.0.0.weight',

        'features.3.conv.1.0.weight',

        'features.3.conv.2.weight',

        'features.4.conv.0.0.weight',

        'features.4.conv.1.0.weight',

        'features.4.conv.2.weight',

        'features.5.conv.0.0.weight',

        'features.5.conv.1.0.weight',

        'features.5.conv.2.weight',

        'features.6.conv.0.0.weight',

        'features.6.conv.1.0.weight',

        'features.6.conv.2.weight',

        'features.7.conv.0.0.weight',

        'features.7.conv.1.0.weight',

        'features.7.conv.2.weight',

        'features.8.conv.0.0.weight',

        'features.8.conv.1.0.weight',

        'features.8.conv.2.weight',

        'features.9.conv.0.0.weight',

        'features.9.conv.1.0.weight',

        'features.9.conv.2.weight',

        'features.10.conv.0.0.weight',

        'features.10.conv.1.0.weight',

        'features.10.conv.2.weight',

        'features.11.conv.0.0.weight',

        'features.11.conv.1.0.weight',

        'features.11.conv.2.weight',

        'features.12.conv.0.0.weight',

        'features.12.conv.1.0.weight',

        'features.12.conv.2.weight',

        'features.13.conv.0.0.weight',

        'features.13.conv.1.0.weight',

        'features.13.conv.2.weight',

        'features.14.conv.0.0.weight',

        'features.14.conv.1.0.weight',

        'features.14.conv.2.weight',

        'features.15.conv.0.0.weight',

        'features.15.conv.1.0.weight',

        'features.15.conv.2.weight',

        'features.16.conv.0.0.weight',

        'features.16.conv.1.0.weight',

        'features.16.conv.2.weight',

        'features.17.conv.0.0.weight',

        'features.17.conv.1.0.weight',

        'features.17.conv.2.weight',

        'features.18.0.weight',


        # 'conv1.weight',
        # 'layer1.0.conv1.weight',
        # 'layer1.0.conv2.weight',
        # 'layer1.0.conv3.weight',
        # 'layer1.0.downsample.0.weight',
        # 'layer1.1.conv1.weight',
        # 'layer1.1.conv2.weight',
        # 'layer1.1.conv3.weight',
        # 'layer1.2.conv1.weight',
        # 'layer1.2.conv2.weight',
        # 'layer1.2.conv3.weight',
        # 'layer2.0.conv1.weight',
        # 'layer2.0.conv2.weight',
        # 'layer2.0.conv3.weight',
        # 'layer2.0.downsample.0.weight',
        #
        # 'layer2.1.conv1.weight',
        # 'layer2.1.conv2.weight',
        # 'layer2.1.conv3.weight',
        # 'layer2.2.conv1.weight',
        # 'layer2.2.conv3.weight',
        # 'layer2.2.conv2.weight',
        # 'layer2.3.conv1.weight',
        # 'layer2.3.conv2.weight',
        # 'layer2.3.conv3.weight',
        # 'layer3.0.conv1.weight',
        # 'layer3.0.conv2.weight',
        # 'layer3.0.conv3.weight',
        # 'layer3.0.downsample.0.weight',
        #
        # 'layer3.1.conv1.weight',
        # 'layer3.1.conv2.weight',
        # 'layer3.1.conv3.weight',
        # 'layer3.2.conv1.weight',
        # 'layer3.2.conv2.weight',
        # 'layer3.2.conv3.weight',
        # 'layer3.3.conv1.weight',
        # 'layer3.3.conv2.weight',
        # 'layer3.3.conv3.weight',
        # 'layer3.4.conv1.weight',
        # 'layer3.4.conv2.weight',
        # 'layer3.4.conv3.weight',
        # 'layer3.5.conv1.weight',
        # 'layer3.5.conv2.weight',
        # 'layer3.5.conv3.weight',
        # 'layer4.0.conv1.weight',
        # 'layer4.0.conv2.weight',
        # 'layer4.0.conv3.weight',
        # 'layer4.0.downsample.0.weight',
        # 'layer4.1.conv1.weight',
        # 'layer4.1.conv2.weight',
        # 'layer4.1.conv3.weight',
        # 'layer4.2.conv1.weight',
        # 'layer4.2.conv2.weight',
        # 'layer4.2.conv3.weight',
    ]
    fc_quan = ['classifier.1.weight']


    # opt = parse.parse_args()
    device='cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mobilenet_v2().to(device)

    model.classifier = nn.Linear(1280, 10)
    model.load_state_dict(torch.load("./mobilenetv2_86_accuracy9667.pt"))
    time_start = time.time()
    with torch.no_grad():
        model.eval()
        for layername in quan_2d:
            for name,param in model.named_parameters():
                if (name == layername):
                    print(name)
                    # print(param)
                    for m in range(len(param)):
                        for k in range(len(param[0])):
                            max = 0
                            min = 0
                            max_obs = 0
                            max_obstmp = 0
                            min_ci_fang = 0
                            # print(param[m][k])
                            max = torch.max(param[m][k])
                            min = torch.min(param[m][k])
                            max_obs = torch.max(max, (-min))
                            while max_obs > 1 / 4096:
                                max_obs >>= 1
                                min_ci_fang += 1

                            param[m][k] = quantify_ten(param[m][k], min_ci_fang)
                            # print("Range:", pow(2, min_ci_fang - 12), "Min_unit:", pow(2, min_ci_fang - 19))
                            # print(param[m][k])
                    #print(param)
        for layername in quan_1d:
            for name, param in model.named_parameters():
                if (name == layername):
                    print(name)
                    min_ci_fang = 0
                    # print(param)
                    for i in range(len(param)):
                        # print(param[i])
                        max_obs = torch.max(param[i], (-param[i]))
                        min_ci_fang = 0
                        while max_obs > 1 / 4096:
                            max_obs >>= 1
                            min_ci_fang += 1
                        param[i] = quantify(param[i], min_ci_fang)
                        # print("Range:", pow(2, min_ci_fang - 12), "Min_unit:", pow(2, min_ci_fang - 19))
                        # print(param[i])
        for layername in quan_pw1d:
            for name, param in model.named_parameters():
                if (name == layername):
                    print(name)
                    min_ci_fang = 0
                    # print(param)
                    for i in range(len(param)):

                        for j in range(len(param[i])):
                            print(param[i][j])
                            max_obs = torch.max(param[i][j], (-param[i][j]))
                            min_ci_fang = 0
                            while max_obs > 1 / 4096:
                                max_obs >>= 1
                                min_ci_fang += 1
                            param[i][j] = quantify(param[i][j], min_ci_fang)
                        # print("Range:", pow(2, min_ci_fang - 12), "Min_unit:", pow(2, min_ci_fang - 19))
                        # print(param[i][j])
        for layername in fc_quan:
            for name,param in model.named_parameters():
                if (name == layername):
                    print(name)
                    # print(param)
                    para_bef = param
                    min_ci_fang = 0
                    for i in range(len(param)):
                        max = 0
                        min = 0
                        max_obs = 0
                        min_ci_fang = 0
                        max = torch.max(param[i])
                        min = torch.min(param[i])
                        max_obs = torch.max(max,(-min))

                        while max_obs > 1 / 4096:
                            max_obs >>= 1
                            min_ci_fang += 1

                        for j in range(len(param[0])):
                            param[i][j] = quantify(param[i][j], min_ci_fang)
                        # print("Range:", pow(2, min_ci_fang - 12), "Min_unit:", pow(2, min_ci_fang - 19))
                    # print(param)
        torch.save(model.state_dict(), './mbv29667.pt')
        time_end = time.time()
        print(time_end - time_start)

