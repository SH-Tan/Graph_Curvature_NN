import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import torch.nn as nn
from collections import defaultdict

import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.integrate import simps
import pickle

from sklearn.linear_model import LinearRegression as lg

import tools.utils as utils
from tools.small_model import FC_MD
from tools.FC_linear import FC_Linear
from tools.LeNet5_small import LeNet as LeNet
from tools.LeNet5_custom_small import LeNet_custom_v2 as LeNet_custom_v2
# from tools.LeNet5_small_linear import LeNet as LeNet
# from tools.LeNet5_custom_small_linear import LeNet_custom_v2 as LeNet_custom_v2

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


data_train = MNIST('./data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))


layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

nodes_num = 2118

model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 28}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 12}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 4}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 84}},
    6: {"name": "fc", "dim": {"out_size": 10}}
}

loss_fn = nn.CrossEntropyLoss()
selected_classes = [0,1,2,3,4,5,6,7,8,9]
    
model_path = "CNN/models/"

res_path = 'img/w7/'
data_path = 'CNN/res/'

# fc_name_auc = "_robust.pkl"
# fc_name_no_auc = "_norobust.pkl"
# fc_name_frac = "frac_robust.pkl"
# fc_name_no_frac = "frac_norobust.pkl"

# fc_name_auc = "_robust_linear.pkl"
# fc_name_no_auc = "_norobust_linear.pkl"
# fc_name_frac = "frac_robust_linear.pkl"
# fc_name_no_frac = "frac_norobust_linear.pkl"

cnn_name_auc = '_robust.pkl'
cnn_name_no_auc = '_norobust.pkl'
cnn_name_frac = '_frac_robust.pkl'
cnn_name_no_frac = '_frac_norobust.pkl'

q_str = "_w7_" 
mark = "_ori_c_"

model_n = "cnn"


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


def test(n, loader, eps, alpha, iters):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)
    sample = 20
    finish = set()
    
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            adv_img = standard_PGD(n, images, labels, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
            
            # adv_img1 = standard_PGD(n, images, labels, 0.05, alpha, iters)
            # adv_out1 = n(adv_img1)
            # adv_pred1 = adv_out1.detach().max(1)[1]
 
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(labels.view_as(pred))
            
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(pred))
   
            succ_pair[l].append((images[succ_l], adv_img[succ_l]))
            robust_pair[l].append((images[robust_l], adv_img[robust_l]))

        # print(f'Finish label {l}....')

    return succ_pair, robust_pair


def single_test(n, im, pgd_im, l, eps, alpha, iters):
    n.eval()
    im = im[np.newaxis,:]
    im = im.to(device)
    output = n(im)
    labels = output.detach().max(1)[1]
    loss_ori = loss_fn(output, labels)
    
    # adv_img = standard_PGD(n, im, labels, eps, alpha, iters)
    pgd_im = pgd_im[np.newaxis,:]
    adv_out = n(pgd_im)
    loss_pgd = loss_fn(adv_out, labels)
            
    loss_delta = loss_pgd - loss_ori
    
    # print(f'Correct label is {l}, predition label is {labels.cpu().item()}, attacked img prediction is {adv_pred.cpu().item()}....\n')
    return loss_delta.cpu().item()



def get_auc(name):
    AUC = []
    with open(data_path + name, 'rb') as file:
        robust_dict = pickle.load(file)
    
    for l in selected_classes:
        for c in robust_dict[l]:
            d = np.sort(c)
            
            ecdf = sm.distributions.ECDF(d)
            x = np.linspace(-50, 1, num=1000)
            y = ecdf(x)
            
            a = simps(y, x, dx=0.001)
            AUC.append(a)
        
    return AUC


def get_fraction(name):
    with open(data_path + name, 'rb') as file:
        robust_dict = pickle.load(file)
        
    frac = []
    
    for l in selected_classes:
        for f in robust_dict[l]:
            frac.append(f)

    return frac



# AUC, loss = zip(*sorted(zip(AUC, loss))) 
def draw(AUC, loss, mark = ''): 
    assert(len(AUC) == len(loss))
    
    X_train = np.array(loss).reshape((len(loss), 1))
    Y_train = np.array(AUC).reshape((len(AUC), 1))
    lineModel = lg()
    lineModel.fit(X_train, Y_train)
 
    Y_predict = lineModel.predict(X_train)
    
    a1 = lineModel.coef_[0][0]
    b = lineModel.intercept_[0]
    
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.scatter(loss, AUC, c = 'skyblue', marker = '*', label='Example')
    ax1.plot(loss, Y_predict, 'r', label='Fitted Line: y = ' + str(round(a1,4)) + '*x + ' + str(round(b,4)))
    
    plt.xlabel('Delta Loss', fontsize = 19, fontweight='semibold')
    plt.ylabel('Negative Curvature Edges Ratio', fontsize = 19, fontweight='semibold')
    plt.yticks(size=18,weight='semibold')
    plt.xticks(size=18,weight='semibold')
    plt.legend(loc = 'best', prop={'size':19, 'weight':'semibold'})
    plt.grid(True)
    
    plt.savefig(res_path + model_n + mark + "_loss.eps")
    plt.close()
    
    return a1
    

    

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
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=1, valid_num=2000)

    sep_valloader = utils.sep_label(valid_dataset, selected_classes, bs=2000)
    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    eps = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    Q = [1]
    
    # build model
    for layer_num in [2]:
        for q in Q:
            # model_name = "best_ori_10l_" + str(layer_num) + ".pth"
            # model_name = "pgdtrain_" + str(layer_num) + ".pth"
            # model_name = "best_ori_" + str(layer_num) + "_linear.pth"
            # model_name = "best_adv_" + str(layer_num) + "_linear.pth"
            
            model_name= "mnist_relu_small.pth"
            # model_name= "mnist_linear_small.pth"
            
            print(f'Now for model {model_name}....\n')
            
            net_H = LeNet_custom_v2(model_dims, None, device)
            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)
            
            # dims = model_zoo[layer_num]
            # net_H = FC_Linear(dims, layer_num)
            # net_H.load_state_dict(torch.load(model_path + model_name))
            # net_H = net_H.to(device)
            

            for e in eps:
                succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40)
                
                # file name
                # auc_name = q_str + str(q) + mark + str(e) + '_' + str(layer_num) + fc_name_auc
                # auc_name_no = q_str + str(q) + mark + str(e) + '_' + str(layer_num) + fc_name_no_auc
    
                # frac_name = str(e) + q_str + str(q) + '_' + str(layer_num) + fc_name_frac
                # frac_name_no = str(e) + q_str + str(q) + '_' + str(layer_num) + fc_name_no_frac
                
                auc_name = q_str + str(q) + mark + str(e) + cnn_name_auc
                auc_name_no = q_str + str(q) + mark + str(e) + cnn_name_no_auc
                frac_name = str(e) + q_str + str(q) + cnn_name_frac
                frac_name_no = str(e) + q_str + str(q) + cnn_name_no_frac

                sample_size = 50
                delta_loss_l = []
                
                for l in selected_classes:                    
                    # i = 0
                    # for (ori_im, adv_im) in robust_pair[l]:

                    #     for index in range(0, len(ori_im)):
                    #         im, pgd_im = ori_im[index], adv_im[index]
                    #         loss = single_test(net_H, im, pgd_im, l, e, alpha=2/255, iters=40)
                    #         delta_loss_l.append(loss)
                            
                    #         i += 1 
                    #         if (i >= sample_size):
                    #             break
                        
                    i = 0
                    for (ori_im, adv_im) in succ_pair[l]:
                        for index in range(0, len(ori_im)):
                            im, pgd_im = ori_im[index], adv_im[index]
                            loss = single_test(net_H, im, pgd_im, l, e, alpha=2/255, iters=40)
                            delta_loss_l.append(loss)
                            
                            i += 1
                            if (i >= sample_size):
                                break
                            
                AUC_robust = get_auc(auc_name)
                frac_robust = get_fraction(frac_name)
                AUC_norobust = get_auc(auc_name_no)
                frac_norobust = get_fraction(frac_name_no)
                                
                AUC = AUC_norobust 
                frac = frac_norobust 
                
                print(f'AUC : {len(AUC)}, Delta loss: {len(delta_loss_l)}, FRAC: {len(frac)}')
                
                # if (len(AUC) > 0):              
                #     draw(AUC, delta_loss_l, mark= str(q) + '_e_' + str(e) + '_l_' + str(layer_num))
                
                with open("./slope.txt", "a+") as f:
                    if (len(frac) > 0):
                        a = draw(frac, delta_loss_l, mark= str(q) + '_e_' + str(e) + '_frac_l_' + str(layer_num))
                        f.write(f'W = {q_str}: For model {model_n}, layer {layer_num}, eps = {e}, the slope is {a:.4f}\n\n')
