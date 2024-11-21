import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from time import sleep
import numpy as np
import random
import os
from tqdm import tqdm

from tools.small_model import FC_MD
from torchattacks import PGD
from cur_loss import CUR_LOSS
from get_all_graphs import get_graph, get_mask

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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


# file_path = "edge_weights/"
# model_path = "models/"
# output_path = "res/thre_test/"

file_path = "res_compress/models/"
model_path = "res_compress/"
output_path = "pgd_comp/"


def get_new_data(l1):
    # selected classes
    train_i1 = torch.tensor([i for i, (_, label) in enumerate(data_train) if label in l1])
    test_i1 = torch.tensor([i for i, (_, label) in enumerate(data_test) if label in l1])
    
    train_index = torch.randperm(len(train_i1))
    valid_dataset = torch.utils.data.Subset(data_train, train_i1[train_index[0:5000]])
    train_dataset = torch.utils.data.Subset(data_train, train_i1[train_index[5000:,]])
    test_dataset = torch.utils.data.Subset(data_test, test_i1)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=2)
    
    return train_loader, test_loader, valid_loader, valid_dataset, test_dataset


def sep_label(dataset, ls):
    sep_dataloader = []
    for l in ls:
        index = torch.tensor([i for i, (_, label) in enumerate(dataset) if label == l])
        subset = torch.utils.data.Subset(dataset, index)
        loader = DataLoader(subset, batch_size=10000, num_workers=2)

        sep_dataloader.append(loader)
        
    return sep_dataloader



# PGD Attack
def pgd_attack(model, images, labels, dims, mask_dict, eps=11/255, alpha=2/255, iters=40, target = None) :
    images = images.to(device)
    labels = labels[0].cpu().item()
    Loss = CUR_LOSS(model, dims, mask_dict[labels], mask_dict, target) # target label: mask
        
    ori_images = images.data
        
    for i in range(iters) :    
        # with tqdm(total=len(images)) as t:
        #     t.set_description('Iteration %i' % i)
        images.requires_grad = True
        # outputs = model(images)

        model.zero_grad()
        cost = Loss(images).to(device)
        cost.backward()

        adv_images = images - alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
            # t.set_postfix(loss=cost)
            # sleep(0.01)
            # t.update(1)
            
    return images

def adv_train(loader, dims, net, mask_dict, thre, sample = 30):
    net.eval()
    ori_acc = 0.
    adv_acc = 0.
    layer_num = len(dims) - 2
    
    target_acc = 0.
    target = None
    
    t = set(selected_classes)
    with open(output_path + "newPGD_full_" + str(layer_num) + ".txt","a+") as f:
        f.write(f'Now using threshold {thre}...')
        f.write('\n')
        for cur_loader in loader:
            for i, (images, labels) in enumerate(cur_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                cur_l = labels[0].cpu().item()
                # target = t - set([cur_l])

                output = net(images)
                pred1 = output.detach().max(1)[1]
                ori_acc += pred1.eq(labels.view_as(pred1)).sum()
                
                adv_img = pgd_attack(net, images, labels, dims, mask_dict, target=target)
                
                output = net(adv_img)
                pred2 = output.detach().max(1)[1]
                adv_acc += pred2.eq(labels.view_as(pred2)).sum()
                
                # for p in pred2:
                #     p = p.cpu().item()
                #     if p in target:
                #         target_acc += 1
                
                # f.write(f'Prediction before attack is {pred1.cpu().item()}, after is {pred2.cpu().item()}, correct label is {labels.cpu().item()}.')
                # f.write('\n')
                # if (i == sample):
                #     print(f'Finish label {labels.cpu().item()}....')
                #     break
        # f.write('\n')
            f.write(f'Current attack label is {cur_l}...')
            f.write('\n')
            ori_acc = float(ori_acc) / len(cur_loader.dataset)
            adv_acc = float(adv_acc) / len(cur_loader.dataset)
            target_acc = float(target_acc) / len(cur_loader.dataset)
            f.write(f'Test for label {labels[0].cpu().item()}  origin Accuracy: {ori_acc:.3f}, adversary ACC: {adv_acc:.3f}, target ACC: {target_acc:.3f}.')
            f.write('\n')
            f.write('\n')
            print(f'Finish label {labels[0].cpu().item()}....')
        print()


def main(sep_dataloader, dims, net, mask_dict, thre):
    adv_train(sep_dataloader, dims, net, mask_dict, thre)


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
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = get_new_data(selected_classes)

    test_sep_dataloader = sep_label(test_dataset, selected_classes)
    
    print(f'Successfully generate separate dataloaders.')
    
    dims = [784, 20, 15, 10]
    layer_num = 2

    model_name = "best_ori_10l_" + str(layer_num) + ".pth"

    net = FC_MD(dims, layer_num)

    net.load_state_dict(torch.load(model_path + model_name))
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)
    
    G_l, edges_num = get_graph(net, dims, selected_classes, file_path) 
    
    # num of selected top edges
    num = 100
    THRESHOLD_MIN = 0.90
    THRESHOLD_MAX = 0.99
    STEP = 0.01
    
    l = np.arange(THRESHOLD_MIN, THRESHOLD_MAX+STEP, STEP)
    
    for thre in [0]:   
        mask_dict = get_mask(G_l, edges_num, dims, num=num, thre=thre)
        main(test_sep_dataloader, dims, net, mask_dict, thre)
