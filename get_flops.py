from cifra_10_cut_net import tinymcunet
from resnet18_quan_net8 import Resnet18
import torchvision
from torchvision.models.resnet import resnet50
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import MobileNetV2
from torch.utils.tensorboard import SummaryWriter
from resnet18yzp import resnet18
from mobilenet_224 import MobileNet_v1
from torchvision.models.mobilenetv2 import MobileNetV2,mobilenet_v2
# from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
import torch.optim as optim
import torch.nn as nn
from PIL import Image,ImageDraw
import torch
from time import sleep
import os
from torchstat import stat
# from scipy.misc import imsave
from imageio import imsave
import time
if __name__ == '__main__':
    all_loss = 0
    all_accNum = 0
    writer = SummaryWriter('runs')
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_datasets = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=True)
    val_dataloader = DataLoader(val_datasets, batch_size=1, shuffle=False)
    device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = tinymcunet().to(device)

    model = mobilenet_v2().to(device)
    # 初始化net,训练和验证都需要net
    # net = mobilenet_v2(pretrained=False)
    for name,param in model.named_parameters():
        print('\'',name,'\',',sep="")

    # 验证修改是否成功
    # changed_dict = torch.load('./model_changed.pth')
    # print(older_val)
    # print(changed_dict['features.0.0.weight'])


    # stat(model, (3, 224, 224))


    # model.eval()
    # all_acc = 0
    # for idx, (img, labels) in enumerate(val_dataloader):
    #     img = img.to(device)
    #     labels = labels.to(device)
    #
    #     out = model(img)
    #     cur_accNum = (out.data.max(dim=1)[1] == labels).sum() / len(labels)
    #     all_acc += cur_accNum
    #     # print('quan:', all_acc / len(val_dataloader), len(val_dataloader))
    #     print('idx:', idx)
    #     print('quan:',all_acc / (idx+1))
    #     # print(all_acc)
    # print(len(val_dataloader))
