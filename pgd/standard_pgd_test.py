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
from cur_loss import CUR_LOSS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.append("..")

from tools.small_model import FC_MD

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
model_path = "models/"
# output_path = "res/thre_test/"

file_path = "res_compress/models/"
res_path = "pgd_comp/"

# full model
dims = [784, 20, 15, 10]
layer_num = 2

model_name = "best_ori_10l_" + str(layer_num) + ".pth"

net_full = FC_MD(dims, layer_num)


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
    sep_dataloader = dict()
    for l in ls:
        index = torch.tensor([i for i, (_, label) in enumerate(dataset) if label == l])
        subset = torch.utils.data.Subset(dataset, index)
        loader = DataLoader(subset, batch_size=5000, num_workers=2)

        sep_dataloader[l] = loader
        
    return sep_dataloader


def get_net_info(net):
    neural_list = []
    nodes_num = 0
    edges_num = 0
    i = 0
    for p in net.parameters():
        if i == 0:
            nodes_num += p.shape[1]
        if i%2 == 0:
            nodes_num += p.shape[0]
            edges_num += (p.shape[0] * p.shape[1])
            neural_list.append(p.shape[0])
        i += 1
        
    print(f'Total nodes are {nodes_num}, total edges are {edges_num}.')
    
    return nodes_num, edges_num


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


def adv_train(loader, dims, net, net_full, mask_dict, thre, mask, f, eps=11/255, alpha=2/255, iters=40):
    net.eval()
    
    real_acc = 0.
    adversary_acc = 0.
    sub_real_acc = 0.
    sub_adversary_acc = 0.
    
    ori_acc = 0.
    adv_acc = 0.
    succ_attack = 0.
    same_attack = 0.
    acc_full = 0.
    layer_num = len(dims) - 2
    
    target_acc = 0.
    target = None
    
    # t = set(selected_classes)
    # with open(res_path + "standardPGD_attack_" + str(layer_num) + ".txt","a+") as f:
    f.write(f'Now using threshold {thre}, the current test model is {mask}...')
    f.write('\n')
    for cur_loader in loader.values():
        for i, (images, labels) in enumerate(cur_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            cur_l = labels[0].cpu().item()
            # target = t - set([cur_l])

            output1 = net(images)
            pred1 = output1.detach().max(1)[1]
            ori_acc += pred1.eq(labels.view_as(pred1)).sum()
            
            output2 = net_full(images)
            pred2 = output2.detach().max(1)[1]
            acc_full += pred2.eq(labels.view_as(pred2)).sum()
            
            # adv_img = pgd_attack(net, images, labels, dims, mask_dict, target=target)
            adv_img = standard_PGD(net, images, labels, eps, alpha, iters)
            
            output3 = net(adv_img)
            pred3 = output3.detach().max(1)[1]
            adv_acc += pred3.eq(labels.view_as(pred3)).sum()
            
            output4 = net_full(adv_img)
            pred4 = output4.detach().max(1)[1]
            same_attack += pred4.eq(pred3.view_as(pred4)).sum()
            succ_attack += pred4.eq(labels.view_as(pred4)).sum()
            
            # for p in pred2:
            #     p = p.cpu().item()
            #     if p in target:
            #         target_acc += 1
            

        # f.write(f'Current attack label is {cur_l}...')
        # f.write('\n')
        ori_acc = float(ori_acc) / len(cur_loader.dataset)
        adv_acc = float(adv_acc) / len(cur_loader.dataset)
        # target_acc = float(target_acc) / len(cur_loader.dataset)
        same_attack = float(same_attack) / len(cur_loader.dataset)
        succ_attack = float(succ_attack) / len(cur_loader.dataset)
        acc_full = float(acc_full) / len(cur_loader.dataset)
        
        # f.write(f'Test for label {labels[0].cpu().item()}  origin Accuracy for full net is: {acc_full:.3f}, for sub net is {ori_acc:.3f}....')
        # f.write('\n')
        # f.write(f'Adversary ACC for full model is: {succ_attack:.3f}, for sub net is {adv_acc:.3f}, Same prediction: {same_attack:.3f}....\n')
        # f.write('\n')
        # print(f'Finish label {labels[0].cpu().item()}....')
        
        real_acc += acc_full
        adversary_acc += succ_attack
        sub_real_acc += ori_acc
        sub_adversary_acc += adv_acc
    print()
    
    f.write(f'Average real accuracy for original full net is {real_acc/10:.3f}, after attack is {adversary_acc/10:.3f}...\n')
    f.write(f'Average real accuracy for sub full net is {sub_real_acc/10:.3f}, after attack is {sub_adversary_acc/10:.3f}...\n')
    
    return real_acc/10, adversary_acc/10, sub_real_acc/10, sub_adversary_acc/10


def main(dataloader, dims, net, net_full, f = None, mask = None, mask_dict = None, thre = None, eps=11/255, alpha=2/255, iters=40):
    acc_full, succ_attack, sub_real_acc, sub_adversary_acc = adv_train(dataloader, dims, net, net_full, mask_dict, thre, mask, f, eps, alpha, iters)
    return acc_full, succ_attack, sub_real_acc, sub_adversary_acc


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
    
    
    net_full.load_state_dict(torch.load(model_path + model_name))
    net_full = net_full.to(device)
    
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = get_new_data(selected_classes)

    test_sep_dataloader = sep_label(test_dataset, selected_classes)
    val_sep_dataloader = sep_label(valid_dataset, selected_classes)
    
    nodes_num, edges_num = get_net_info(net_full)
    
    net_pos = FC_MD(dims, layer_num)
    net_neg = FC_MD(dims, layer_num)
    model_n = "full.pth"
    net_pos.load_state_dict(torch.load(file_path + model_n))
    net_pos = net_pos.to(device)
    
    with open(res_path + "oldPGD_attack_Allsmall" + str(layer_num) + ".txt","a+") as f:
        f.write(f'Full network has {edges_num} edges...\n')
        # for l in selected_classes:
            # f.write(f'Now net is for class {l}....\n')
            # loader = val_sep_dataloader[l]
        # model_n = "class_" + str(l) + ".pth"
        
        # net_pos.load_state_dict(torch.load(file_path + model_n))
        # net_pos = net_pos.to(device)

        # file_n = "less_edge_v" + str(l) + ".csv"
        # G1 = sep_net(net_full, net_pos, dims, layer_num, loader, nodes_num, res_path, device)
        
        net_full.load_state_dict(torch.load(model_path + model_name))
        net_pos.load_state_dict(torch.load(file_path + model_n))
    
        acc_full, succ_attack, sub_real_acc, sub_adversary_acc = main(val_sep_dataloader, dims, net_pos, net_full, f, mask = "Full 10 labels")
        
        # f.write('*'*50 + 'Complement net' + '*'*50)
        # f.write('\n')
        # f.write('\n')
        
        # net_neg = net_neg.to(device)
        # net_neg.load_state_dict(torch.load(res_path + "complement.pth"))
        
        # main(test_sep_dataloader, dims, net_neg, net_full, f, mask = "complement net")
        
        f.write('='*100)
        f.write('\n')
        f.write('\n')
            
            # pos_G_l, pos_edges_num = get_graph(net_pos, dims, selected_classes, file_path) 
            # neg_G_l, neg_edges_num = get_graph(net_neg, dims, selected_classes, file_path) 
