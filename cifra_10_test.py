from mobilenetv1_cut_net import MobileNet_v1
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# from cifra_nano_net import tinymcunet
from cifra_10_cut_net import  tinymcunet
import torch.optim as optim
import torch.nn as nn
from PIL import Image,ImageDraw
import torch
import cv2
from time import sleep
import os
# from scipy.misc import imsave
from imageio import imsave
import time
from torchsummary import summary
if __name__ == '__main__':

    writer = SummaryWriter('runs')
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_datasets = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=True)
    val_dataloader = DataLoader(val_datasets, batch_size=32, shuffle=False)
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tinymcunet().to(device)
    summary(model, (3, 64, 64))
    model.load_state_dict(torch.load("./tinymcunet_quan.pt"))
    model.eval()

    img = cv2.imread('test2.png')
    new_img = torch.tensor(img).permute(2, 0, 1)
    # print(len(new_img),len(new_img[0]),len(new_img[0][0]))
    new_img = torch.unsqueeze(new_img, dim=0) / 255
    result = model(new_img.to(device))
    print(result)
    #print(list(model.parameters()))
    # for name,para in model.named_parameters():
    #     print("'"+name+"',")  # 只查看形状
        #print(para)