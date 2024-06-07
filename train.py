from mobilenet_224 import  MobileNet_v1
from resnet18_cut_net import Resnet18
from torch.utils.tensorboard import SummaryWriter
from torch import nn,optim
import argparse
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
import torch.nn as nn
import torch

DEVICE ='cuda'

class Train:
    def __init__(self, root):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.summaryWriter = SummaryWriter('logs')
        self.train_datasets = torchvision.datasets.CIFAR10('data', train=True, transform=transform_train, download=True)
        self.val_datasets = torchvision.datasets.CIFAR10('data', train=False, transform=transform_test, download=True)
        self.train_dataloader = DataLoader(self.train_datasets, batch_size=32, shuffle=True, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_datasets, batch_size=32, shuffle=False, pin_memory=True)

        self.net_test = MobileNet_v1().to(DEVICE)

        self.opt = optim.Adam(self.net_test.parameters(),lr=0.0001)
        self.img_loss_fun = nn.CrossEntropyLoss()  # 分类问题

        self.train = True
        self.test = False

    def __call__(self):
        # writer = SummaryWriter('runs')
        for epoch in range(1000):
            if self.train:
                all_accNum = 0
                self.net_test.train()
                for idx, (img, labels) in enumerate(self.train_dataloader):
                    img = img.to(DEVICE)
                    labels = labels.to(DEVICE)
                    out = self.net_test(img)

                    los = self.img_loss_fun(out,labels)
                    self.opt.zero_grad()
                    los.backward()
                    self.opt.step()

                    cur_acc = (out.data.max(dim=1)[1] == labels).sum()
                    all_accNum += cur_acc
                    # 每prinft输出一次训练效果
                    if (idx % 300) == 0:
                        print('epoch:{} training:[{}/{}] loss:{:.6f} accuracy:{:.6f}% lr:{}'.format(epoch, idx,len(self.train_dataloader),los.item(),cur_acc * 100 / len(labels),self.opt.param_groups[0]['lr']))
                # writer.add_scalar('accuracy', cur_acc * 100 / len(labels), epoch)

                acc = all_accNum * 100 / (len(self.train_dataloader) * 32)  # batch_size=32
                print('epoch:{} tra_accuracy:{:.6f}%'.format(epoch, acc))

                self.net_test.eval()
                all_accNum = 0
                for idx, (img, labels) in enumerate(self.val_dataloader):
                    img = img.to(DEVICE)
                    labels = labels.to(DEVICE)
                    out = self.net_test(img)
                    cur_acc = (out.data.max(dim=1)[1] == labels).sum()/len(labels)
                    all_accNum += cur_acc
                acc = all_accNum * 100 / (len(self.val_dataloader))  # batch_size=32
                print('epoch:{} val_accuracy:{:.6f}%'.format(epoch, acc))
                if (epoch % 10 == 0):
                    torch.save(self.net_test.state_dict(), f'model/mbv1_{epoch}_accuracy{acc}.pt')

if __name__ == '__main__':
    train = Train('yzp')
    train()