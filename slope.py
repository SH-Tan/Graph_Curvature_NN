import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
import torchvision
import numpy as np
import random
import os
import torch.nn as nn
from collections import defaultdict

import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression as lg

import tools.utils as utils
from tools.small_model import FC_MD
from tools.FC_linear import FC_Linear
from tools.LeNet5_custom_small import LeNet_custom_v2 as LeNet_custom_v2
from tools.LeNet5_custom import LeNet_custom
from tools.cnn_costom import CNN_custom

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


mnist_train = MNIST('./data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

mnist_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

transform_train = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = torchvision.transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_train = CIFAR10('./data/cifar10', train=True, download=True, transform=transform_train)
cifar_test = CIFAR10('./data/cifar10', train=False, download=True, transform=transform_test)


layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    21: [784, 200, 150, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

nodes_num = 2118

mnist_model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 28}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 12}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 4}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 84}},
    6: {"name": "fc", "dim": {"out_size": 10}}
}

cifar_model_dims = {
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 14}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 5}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 84}},
    6: {"name": "fc", "dim": {"out_size": 10}}
}

cifar_bigmodel_dims = {
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},
    2: {"name": "cnn", "dim": {"channel": 32, "kernel": 6, "stride": 2, "out_size": 14}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 5}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 10}}
}

loss_fn = nn.CrossEntropyLoss()
selected_classes = [0,1,2,3,4,5,6,7,8,9]
    



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
        # images = (ori_images + eta).detach_()
            
    return images


def test(n, loader, eps, alpha, iters):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)

    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            adv_img = standard_PGD(n, images, labels, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
 
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(labels.view_as(pred))
            
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(pred))
   
            succ_pair[l].append((images[succ_l], adv_img[succ_l]))
            robust_pair[l].append((images[robust_l], adv_img[robust_l]))

    return succ_pair, robust_pair


def single_test(n, im, pgd_im, l, eps, alpha, iters):
    n.eval()
    im = im[np.newaxis,:]
    im = im.to(device)
    output = n(im)
    labels = output.detach().max(1)[1]
    loss_ori = loss_fn(output, labels)
    
    pgd_im = pgd_im[np.newaxis,:]
    adv_out = n(pgd_im)
    loss_pgd = loss_fn(adv_out, labels)
            
    loss_delta = loss_pgd - loss_ori
    
    # print(f'Correct label is {l}, predition label is {labels.cpu().item()}, attacked img prediction is {adv_pred.cpu().item()}....\n')
    return loss_delta.cpu().item()




def get_fraction(name, data_path):
    with open(data_path + name, 'rb') as file:
        robust_dict = pickle.load(file)
        
    frac = []
    
    for l in selected_classes:
        for f in robust_dict[l]:
            frac.append(f)

    return frac



# AUC, loss = zip(*sorted(zip(AUC, loss))) 
def draw(frac, loss, res_path, mark = ''): 
    assert(len(frac) == len(loss))
    
    X_train = np.array(loss).reshape((len(loss), 1))
    Y_train = np.array(frac).reshape((len(frac), 1))
    lineModel = lg()
    lineModel.fit(X_train, Y_train)
 
    Y_predict = lineModel.predict(X_train)
    
    a1 = lineModel.coef_[0][0]
    b = lineModel.intercept_[0]
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(loss, frac, c = 'skyblue', marker = '*', label='Example')
    ax1.plot(loss, Y_predict, 'r', label='Fitted Line: y = ' + str(round(a1,4)) + '*x + ' + str(round(b,4)))
    
    plt.xlabel('Delta Loss', fontsize = 23, fontweight='semibold')
    plt.ylabel('Negative Curvature Edges Ratio', fontsize = 23, fontweight='semibold')
    plt.yticks(size=22,weight='semibold')
    plt.xticks(size=22,weight='semibold')
    plt.legend(loc = 'best', prop={'size':22, 'weight':'semibold'})
    plt.grid(True)
    
    plt.savefig(res_path + mark + "_loss.eps")
    plt.close()
    
    return a1
    

    

def cal_slope(args):
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    eps = [0.03, 0.07, 0.1, 0.2]
    Q = [1]
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    data_path = args.mnist_data_path
    model_path = args.model_path
    metric = args.metric
    sample_size = args.sample_num
    dataset = args.dataset
    cifar = args.cifar
    
    if dataset.lower() == 'mnist':
        data_train = mnist_train
        data_test = mnist_test
    elif dataset.lower() == 'cifar':
        data_train = cifar_train
        data_test = cifar_test
        eps = [1,2,3,5]
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=1, valid_num=2000)

    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    layers = [2]
    if model_type.lower() == "fc":
        layers = [2,4]
             
    norobust_suffix = "frac_norobust.pkl"
    robust_suffix = "frac_robust.pkl"
    
    # build model
    for layer_num in layers:
        dims = model_zoo[layer_num]
        # fc model
        if model_type.lower() == "fc":
            if model_pre_name.lower() == 'big':
                model_name = "best_21_adv.pth"
                dims = model_zoo[21]
                norobust_suffix = "cifar" + "frac_norobust.pkl"
                model_full_n =  model_type.lower() + "ori"
            else:
                if model_pre_name.lower() == "ori" or model_pre_name.lower() == "decay":
                    model_name = "best_ori_10l_" + str(layer_num) + ".pth"
                elif model_pre_name.lower() == "adv":
                    model_name = "pgdtrain_" + str(layer_num) + ".pth"
                else:
                    raise Exception("Invalid model name, model name should be {ori, decay, adv}!")
                
                norobust_suffix = dataset + "frac_norobust.pkl"
                
            net_H = FC_MD(dims, layer_num)
            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)
        
        # fc linear model
        elif model_type.lower() == "fc_linear":
            if model_pre_name.lower() == "ori" or model_pre_name.lower() == "decay":
                model_name = "best_ori_" + str(layer_num) + "_linear.pth"
            elif model_pre_name.lower() == "adv":
                model_name = "best_adv_" + str(layer_num) + "_linear.pth"
            else:
                raise Exception("Invalid model name, model name should be {ori, decay, adv}!")
            
            dims = model_zoo[layer_num]
            net_H = FC_Linear(dims, layer_num)
            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)
            
            norobust_suffix = "frac_norobust_linear.pkl"
        
        # cnn model
        elif model_type.lower() == "cnn" and dataset.lower() == 'mnist':
            model_name= "mnist_relu_small.pth"
            
            net_H = LeNet_custom_v2(mnist_model_dims, None, device)
            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)
            
            norobust_suffix = "frac_norobust_cnn.pkl"
            
        elif model_type.lower() == "cnn" and dataset.lower() == 'cifar':
            if cifar.lower() == 'small':
                model_name= "best_cifar.pth"
                
                if model_pre_name.lower() == "adv":
                    model_name = "best_cifar_adv.pth"
                    
                net_H = LeNet_custom(cifar_model_dims, device, input_c=3)
                norobust_suffix = "frac_norobust_cifar.pkl"
                robust_suffix = "frac_robust_cifar.pkl"
                
            elif cifar.lower() == 'big':
                model_name= "best_cifar_adv_cnn.pth"
                net_H = CNN_custom(cifar_bigmodel_dims, device, input_c=3)
                norobust_suffix = "frac_norobust_cifar_big.pkl"
            else:
                raise Exception("Invalid CIFAR model type, model type should be {small, big}!")
                
            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)

        print(f'Now for model {model_name}....\n')
        
        for q in Q: 
            for e in eps:
                if dataset.lower() == 'cifar':
                    ep = e/255
                else:
                    ep = e
                succ_pair, robust_pair = test(net_H, sep_dataloader, eps=ep, alpha=2/255, iters=40)
                
                frac_name_no = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + norobust_suffix
                
                if model_type.lower() == "cnn":
                    frac_name_no = model_full_n + str(e) + metric + str(q) + norobust_suffix
                    # frac_name = model_full_n + str(e) + metric + str(q) + robust_suffix

                delta_loss_l = []
                
                for l in selected_classes:                    
                    i = 0
                    for (ori_im, adv_im) in succ_pair[l]:
                        for index in range(0, len(ori_im)):
                            im, pgd_im = ori_im[index], adv_im[index]
                            loss = single_test(net_H, im, pgd_im, l, e, alpha=2/255, iters=40)
                            delta_loss_l.append(loss)
                            
                            i += 1
                            if (i >= sample_size):
                                break
                            
                # for l in selected_classes: 
                #     i = 0 
                #     for (ori_im, adv_im) in robust_pair[l]:
                #         for index in range(0, len(ori_im)):
                #             im, pgd_im = ori_im[index], adv_im[index]
                #             loss = single_test(net_H, im, pgd_im, l, e, alpha=2/255, iters=40)
                #             delta_loss_l.append(loss)
                            
                #             i += 1
                #             if (i >= sample_size):
                #                 break
                            
                frac_norobust = get_fraction(frac_name_no, data_path)
                # frac_robust = get_fraction(frac_name, data_path)
                                
                frac = frac_norobust
                
                print(f'Delta loss: {len(delta_loss_l)}, FRAC: {len(frac)}')
                
                with open(res_path + "slope.txt", "a+") as f:
                    if (len(frac) > 0):
                        a = draw(frac, delta_loss_l, res_path, mark= model_full_n + str(q) + '_e_' + str(e) + '_frac_l_' + str(layer_num)+cifar)
                        f.write(f'W = {metric}: For model {model_type} - {model_pre_name}, layer {layer_num}, eps = {e}, the slope is {a:.4f}\n\n')
