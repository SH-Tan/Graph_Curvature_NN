import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx


from tools.small_model_relu import FC_MD


def plot_acc(acc_1, acc_2, name, path, acc_3 = None, acc_4 = None, model_size = None, avg_w = None, avg_c = None):
    fig, ax = plt.subplots()
    
    x = range(len(acc_1))
    x2 = range(len(acc_2))

    max_n = max(max(max(acc_1), max(acc_2)), 0)
    # plot curves
    ax.plot(x, acc_1, label="ori acc", color = 'red', linewidth = 1)
    ax.plot(x2, acc_2, label="ori adv acc", color='green', linewidth = 1)
    
    if (acc_3):
        ax.plot(x, acc_3, label="sub acc", color = 'blue', linewidth = 1)
    if (acc_4):
        ax.plot(x, acc_4, label= "sub adv acc", color = 'cyan', linewidth = 1)
    if model_size:
        ax.plot(x, model_size, label="Model size", color='yellow', linewidth = 2)
    
    ax.set_ylabel('Number', fontsize = 12)
    ax.set_xlabel('Sample', fontsize = 12)
    
    ax.set_xlim(0, len(acc_1))
    ax.set_ylim(0, max_n + 0.1)
    
    x_ticks = np.arange(0, len(acc_1), 1)
    y_ticks = np.arange(0, max_n + 0.1, 0.05)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax2 = ax.twinx() 
    if avg_w:   
        x1 = range(len(avg_w))
        ax2.plot(x1, avg_w, label="Weights variance", color='black', linewidth = 2)
        ax2.set_ylabel('Variance', fontsize = 12)
        ax2.set_ylim(0, max(avg_w) + 20)
        y2_ticks = np.arange(0, max(avg_w) + 20, 100)
        ax2.set_yticks(y2_ticks)
        
    if avg_c:
        ax2.plot(x, avg_c, label="Curvature variance", color='gray', linewidth = 2)
        # ax2.set_ylim(min(avg_c)-1.0, 0)

        # y2_ticks = np.arange(min(avg_c)-1.0, 0, 0.01)
        # ax2.set_yticks(y2_ticks)
        
    ax2.legend()

    ax.set_title(name, fontsize=14)
    ax.legend()

    plt.savefig(path + name + ".png")
    plt.close()
    
    
def remove_e(adj_m, edge_set):
    for (n1, n2) in edge_set:
        adj_m[n1, n2] = 0
    return adj_m



def build_adjm(loader, net, nodes_num, dims, device, flag = False):
    eps = 10e9
    
    for i, (img, label) in enumerate(loader):    
        img = img.to(device)
        edge_array, nodes = net.edge_w_batch(img)
        edge_array = edge_array.cpu().detach().numpy() 
        output = net.get_weights(img)
        output = output.cpu().detach().numpy() 
        
        output[edge_array == 0] = 0.
        
        if (i == 0):
            data = output
        else:
            data = np.vstack((data,output))
    
    w_avg = np.mean(data, axis=0) # (edge num,)
    w_avg = np.abs(w_avg) # absolate edge value for one image
    
    print(f'Tha max weight is {np.max(w_avg)}, min is {np.min(w_avg)}....')
            
    # build adjacent matrix
    adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)

    d_num = len(dims)
    cur_s_col = dims[0]
    cur_e_col = dims[0] + dims[1]

    cur_layer = 1
    start_col = 0
    end_col = dims[1]

    for i in range(nodes_num - dims[d_num-1]):
        # print(f'i : {i}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
        adjacent_m[i, cur_s_col : cur_e_col] = w_avg[start_col : end_col]
        
        if (cur_layer < d_num-1 and i == cur_s_col - 1):
            cur_layer += 1
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            cur_s_col = cur_e_col
            cur_e_col = cur_e_col + dims[cur_layer]
        else:
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            
    # ind = np.where(adjacent_m < 0)

    # i = ind[0]
    # j = ind[1]

    # adjacent_m[j,i] = -adjacent_m[i,j]
    # adjacent_m[adjacent_m<0] = 0
            
    return adjacent_m



def get_new_data(l1, data_train, data_test, train_bs = 128, test_bs = 2000, valid_num = 2500):
    # selected classes
    train_i1 = torch.tensor([i for i, (_, label) in enumerate(data_train) if label in l1])
    test_i1 = torch.tensor([i for i, (_, label) in enumerate(data_test) if label in l1])
    
    train_index = torch.randperm(len(train_i1))
    valid_dataset = torch.utils.data.Subset(data_train, train_i1[train_index[0:valid_num]])
    train_dataset = torch.utils.data.Subset(data_train, train_i1[train_index[valid_num:,]])
    test_dataset = torch.utils.data.Subset(data_test, test_i1)
    
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_bs, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=test_bs, num_workers=2)
    
    return train_loader, test_loader, valid_loader, valid_dataset, test_dataset


def sep_label(dataset, ls, bs = 5000):
    sep_dataloader = dict()
    for l in ls:
        index = torch.tensor([i for i, (_, label) in enumerate(dataset) if label == l])
        subset = torch.utils.data.Subset(dataset, index)
        loader = DataLoader(subset, batch_size=bs, num_workers=2)

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



def test(n, loader, device):
    n.eval()
    total_correct = 0
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    # print(f'Test Accuracy for label {l}: {(float(total_correct) / len(loader.dataset)):.3f}')
    
    acc = float(total_correct) / len(loader.dataset)
    return acc


def standard_PGD(model, images, labels, device, eps=.1, alpha=.1, iters=100):
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


def standard_PGD_test(loader, net, device, eps=.1, alpha=.1, iters=100):
    net.eval()
    
    ori_acc = 0.
    adv_acc = 0.
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        output1 = net(images)
        pred1 = output1.detach().max(1)[1]
        ori_acc += pred1.eq(labels.view_as(pred1)).sum()
        
        # adv_img = pgd_attack(net, images, labels, dims, mask_dict, target=target)
        adv_img = standard_PGD(net, images, labels, device, eps, alpha, iters)
        
        output3 = net(adv_img)
        pred3 = output3.detach().max(1)[1]
        adv_acc += pred3.eq(labels.view_as(pred3)).sum()
        

    ori_acc = float(ori_acc) / len(loader.dataset)
    adv_acc = float(adv_acc) / len(loader.dataset)

    return adv_acc


