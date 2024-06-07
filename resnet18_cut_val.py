import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from resnet18_cut_net import Resnet18
import torch
if __name__ == '__main__':
    all_loss = 0
    all_accNum = 0
    writer = SummaryWriter('runs')
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_datasets = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=True)
    val_dataloader = DataLoader(val_datasets, batch_size=32, shuffle=False)
    device = 'cpu'
    model = Resnet18().to(device)
    model.load_state_dict(torch.load("./resnet18_130.pt"))
    model.eval()
    all_acc = 0
    for idx, (img, labels) in enumerate(val_dataloader):
        img = img.to(device)
        labels = labels.to(device)
        print('idx:',idx)
        out = model(img)
        cur_accNum = (out.data.max(dim=1)[1] == labels).sum() / len(labels)
        all_acc += cur_accNum
    print(all_acc / len(val_dataloader),len(val_dataloader))