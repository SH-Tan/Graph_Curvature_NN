import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os

import sys
sys.path.append("..")

import tools.utils as utils
# from tools.LeNet5_small import LeNet
from tools.LeNet5_small_linear import LeNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



data_train = MNIST('../data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))


model_path = "models/"

criterion = nn.CrossEntropyLoss()


def train(loader, net_H, optimizer):
    net_H.train()

    correct = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)

        # clear up gradients for backprop
        optimizer.zero_grad()
        output = net_H(data)
        
        loss = criterion(output, target)
        total_loss += loss

        # compute gradients and make updates
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())
        
    print('Test set: Avg. Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))))
    


def test(net, test_loader):
    net.eval()
    total_correct = 0
    
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        
    acc = float(total_correct) / len(test_loader.dataset)
    
    print('Test set: Avg. Accuracy: {}/{} ({:.2f}%)'.format(
        total_correct, len(test_loader.dataset),
        (100. * total_correct / len(test_loader.dataset))))
    
    return acc



if __name__ == '__main__':
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    train_loader, test_loader, valid_loader, _, _ = utils.get_new_data(selected_classes, data_train, data_test, valid_num=5000)
    
    model = LeNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1.5e-3)
        
    epoches = 10
    best_acc = 0.
    
    for e in range(epoches):
        train(train_loader, model, optimizer)
        
        acc_t = test(model, valid_loader)
        
        if (acc_t > best_acc):
            best_acc = acc_t
            torch.save(model.state_dict(), model_path + "mnist_linear_small.pth")
    