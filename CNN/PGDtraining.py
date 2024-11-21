import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import copy

import sys
sys.path.append("..")

import tools.utils as utils
from tools.LeNet5 import LeNet as LeNet
from tools.LeNet5_custom_v2 import LeNet_custom_v2 as LeNet_custom_v2


os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



data_train = MNIST('../data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))


out_path = "advtrain/models/"
model_path = "models/"
res_path = "advtrain/"

nodes_num = 9118

model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 32}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 5, "stride": 1, "out_size": 28}},
    3: {"name": "pooling", "dim": {"channel": 6, "kernel": 2, "stride": 2, "out_size": 14}},
    4: {"name": "cnn", "dim": {"channel": 16, "kernel": 5, "stride": 1, "out_size": 10}},
    5: {"name": "pooling", "dim": {"channel": 16, "kernel": 2, "stride": 2, "out_size":5}},
    6: {"name": "fc", "dim": {"out_size": 120}},
    7: {"name": "fc", "dim": {"out_size": 84}},
    8: {"name": "fc", "dim": {"out_size": 10}}
}

criterion = nn.CrossEntropyLoss()


def standard_PGD(model, images, labels, eps=.1, alpha=.1, iters=100):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images


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

    return total_loss/len(loader), correct / len(loader.dataset)
    


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



def test_adversarial(net, loader, eps=.1, alpha=.1, iters=100):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    correct = 0

    for data, target in loader:

        data = standard_PGD(net, data, target, eps, alpha, iters)
        data, target = data.to(device), target.to(device)

        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
    
    print('Test set: Avg. Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))))
    
    return correct / len(loader.dataset)


def train_adversarial(net, net_attack, loader, optimizer, epoch, eps=.1, alpha=.1, iters=100):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()

    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        #print(data.size())

        data = standard_PGD(net_attack, data, target, eps, alpha, iters).to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
        
        loss = criterion(output, target)
        total_loss += loss

        # compute gradients and make updates
        loss.backward()
        optimizer.step()
        
    print('Adversary training set: Avg. Accuracy: {}/{} ({:.2f}%)'.format(
    correct, len(loader.dataset),
    (100. * correct / len(loader.dataset))))

    return total_loss/len(loader), correct / len(loader.dataset)



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
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test)

    # valid_dataloader = utils.sep_label(valid_dataset, selected_classes)
    # sep_dataloader = utils.sep_label(test_dataset, selected_classes)
    
    
    # build model
    # origin model
    model = LeNet()
    model = model.to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epoches = 10
    
    # train
    with open(res_path + "adv training_lenet.txt", "w+") as f:   
        
        for epoch in range(epoches):
            print(f'For epoch {epoch}.....')
            f.write(f'This is Epoch {epoch}...\n')
            
            f.write(f'Traing use train net to attack...\n')
            loss_oldadv, acc_oldadv = train_adversarial(model, model, train_loader, optimizer, epoch, eps=.1, alpha=.1, iters=100)
            
            tmp = copy.deepcopy(model)
            print(f'The loss for adversary training is {loss_oldadv:.3f}, accuracy is {acc_oldadv:.3f}...')     
            # normal test -- validation set
            acc_oldadv = test_adversarial(model, valid_loader, eps=.1, alpha=.1, iters=100)
            f.write(f'The accuracy for adversary training on validation set is {acc_oldadv:.3f}...\n\n')
            
        # adversary test
        f.write('\n')
        f.write('*'*50 + 'adversary test' + '*'*50)
        f.write('\n')
        f.write('\n')
        
        # acc1 = test_adversarial(net_advtrain, test_loader)
        # f.write(f'The adversary test acc on test set for new adversary training model is {acc1:.3f}\n')
        # acc11 = test_adversarial(net_attack, test_loader)
        # f.write(f'The adversary test acc on test set for new adversary attack model is {acc11:.3f}\n\n')
            
        acc2 = test_adversarial(model, test_loader, eps=.1, alpha=.1, iters=100)
        f.write(f'The adversary test acc on test set for old adversary training model is {acc2:.3f}\n\n')
        
        torch.save(model.state_dict(), out_path + "pgdtrain_lenet.pth")
        
        # acc3 = test_adversarial(net_train, test_loader)
        # f.write(f'The adversary test acc on test set for normal training model is {acc3:.3f}\n')
        
        f.write('\n')
        f.write('='*50 + 'FINISH' + '='*50)
        f.write('\n')
        f.write('\n')
        
        
        