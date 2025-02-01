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
from tools.small_model import FC_MD
from RicciCurvature.OllivierRicci import OllivierRicci
from tools.FC_linear import FC_Linear

import standard_pgd_test
import copy

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


out_path = "models/"
model_path = "models/"
res_path = "models/"

layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

criterion = nn.CrossEntropyLoss()


def standard_PGD(model, images, labels, eps=11/255, alpha=2/255, iters=40):
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


def train(loader, net_H):
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



def test_adversarial(net, loader):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    correct = 0

    for data, target in loader:

        data = standard_PGD(net, data, target, eps=.1, alpha=.1, iters=100)
        data, target = data.to(device), target.to(device)

        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
    
    print('Test set: Avg. Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))))
    
    return correct / len(loader.dataset)


def train_adversarial(net, net_attack, loader, optimizer, epoch):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()

    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        #print(data.size())

        data = standard_PGD(net_attack, data, target, eps=.1, alpha=.1, iters=100).to(device)
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
    for layer_num in [2]:
        # model_name = "full_" + str(layer_num) + ".pth"

        dims = model_zoo[layer_num]
        # net_attack = FC_MD(dims, layer_num)
        # net_attack = net_attack.to(device)
        
        # net_train = FC_MD(dims, layer_num)
        # net_train = net_train.to(device)
        
        # optimizer = optim.Adam(net_train.parameters(), lr=1e-3)
        
        # net_advtrain = FC_MD(dims, layer_num)
        # net_advtrain = net_advtrain.to(device)
        
        # optimizer_adv = optim.Adam(net_advtrain.parameters(), lr=1e-3)
        
        net_oldadvtrain = FC_Linear(dims, layer_num)
        net_oldadvtrain = net_oldadvtrain.to(device)
        
        optimizer_oldadv = optim.Adam(net_oldadvtrain.parameters(), lr=1e-3)
        
        epoches = 10
        
        # train
        with open(res_path + "adv training_" + str(layer_num) + ".txt", "w+") as f:   
            # f.write(f'Now is model {model_name}...\n\n')
            # f.write(f'Traing use sub net to attack...\n')
            
            for epoch in range(epoches):
                print(f'For epoch {epoch}.....')
                f.write(f'This is Epoch {epoch}...\n')
                
                # sub_name = utils.compress_net(net_advtrain, valid_loader, dims, layer_num, device, model_name, model_path, out_path, eps=.1, alpha=.1, iters=100)
                
                # net_attack.load_state_dict(torch.load(out_path + sub_name))
                
                # loss_adv, acc_adv = train_adversarial(net_advtrain, net_attack, train_loader, optimizer_adv, epoch)
                # print(f'The loss for adversary training is {loss_adv:.3f}, accuracy is {acc_adv:.3f}...')     
                # f.write(f'The loss for adversary training is {loss_adv:.3f}, accuracy is {acc_adv:.3f}...\n')  
                # # normal test -- validation set
                # acc_adv = test(net_advtrain, valid_loader)
                # f.write(f'The accuracy for adversary training on validation set is {acc_adv:.3f}...\n\n')
                
                f.write(f'Traing use train net to attack...\n')
                loss_oldadv, acc_oldadv = train_adversarial(net_oldadvtrain, net_oldadvtrain, train_loader, optimizer_oldadv, epoch)
                
                # tmp = copy.deepcopy(net_oldadvtrain)
                # negative_edge, final_G = utils.compress_net(tmp, valid_loader, dims, layer_num, device, model_name, model_path, out_path, eps=.1, alpha=.1, iters=100)
                # f.write(f'The negative curvature edges num for current net is {negative_edge}, after compression the graph is {final_G}...\n')
                
                # print(f'The loss for adversary training is {loss_oldadv:.3f}, accuracy is {acc_oldadv:.3f}...')     
                # normal test -- validation set
                acc_oldadv = test(net_oldadvtrain, valid_loader)
                f.write(f'The accuracy for adversary training on validation set is {acc_oldadv:.3f}...\n\n')
                
                # f.write(f'No attack...\n')
                # loss, acc = train(train_loader, net_train)
                # print(f'The loss for normal training is {loss:.3f}, accuracy is {acc:.3f}...')
                # acc = test(net_train, valid_loader)
                # f.write(f'The accuracy for normal training on validation set is {acc:.3f}...\n\n')
                
            # adversary test
            f.write('\n')
            f.write('*'*50 + 'adversary test' + '*'*50)
            f.write('\n')
            f.write('\n')
            
            # acc1 = test_adversarial(net_advtrain, test_loader)
            # f.write(f'The adversary test acc on test set for new adversary training model is {acc1:.3f}\n')
            # acc11 = test_adversarial(net_attack, test_loader)
            # f.write(f'The adversary test acc on test set for new adversary attack model is {acc11:.3f}\n\n')
             
            acc2 = test_adversarial(net_oldadvtrain, test_loader)
            f.write(f'The adversary test acc on test set for old adversary training model is {acc2:.3f}\n\n')
            
            torch.save(net_oldadvtrain.state_dict(), out_path + "pgdtrain_" + str(layer_num) + ".pth")
            
            # acc3 = test_adversarial(net_train, test_loader)
            # f.write(f'The adversary test acc on test set for normal training model is {acc3:.3f}\n')
            
            f.write('\n')
            f.write('='*50 + 'FINISH' + '='*50)
            f.write('\n')
            f.write('\n')
            
            
            