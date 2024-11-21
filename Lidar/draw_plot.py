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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

safety = {
    "DDPG_64_1": "0/10",
    "DDPG_64_2": "2/10",
    "DDPG_64_3": "8/10",
    "DDPG_128_1": "8/10",
    "DDPG_128_2": "4/10",
    "DDPG_128_3": "0/10",
    "TD3_64_1": "9/10",
    "TD3_64_2": "9/10",
    "TD3_64_3": "9/10",
    "TD3_128_1": "9/10",
    "TD3_128_2": "5/10",
    "TD3_128_3": "9/10"    
}


metric = "w7"
data_path = "res/" + metric + "/"

res_path = "img/" + metric + "/"

mark = "_controller_"

types = ['DDPG', 'TD3']
cs = ['1','2','3']
szs = ['64','128']

markers = ['.', 'o', '*', '+', '>', 'v']
Keys =  ['DDPG_128_1', 'DDPG_128_2', 'DDPG_128_3', 'DDPG_64_1', 'DDPG_64_2', 'DDPG_64_3', 'TD3_128_1', 'TD3_128_2', 'TD3_128_3', 'TD3_64_1', 'TD3_64_2', 'TD3_64_3']

def get_fraction(name, controller_n):
    with open(data_path + name, 'rb') as file:
        data = pickle.load(file)
        
    return data[controller_n]



# AUC, loss = zip(*sorted(zip(AUC, loss))) 
# legend: safety of controller
def draw(frac, c_n_l, s): 
    assert(len(frac) == len(c_n_l))
    
    fig1, ax1 = plt.subplots()
    
    for i in range(len(frac)):
        avg = round(np.mean(frac[i]),4)
        ax1.plot(range(len(frac[i])), frac[i], marker = markers[i%6], label=c_n_l[i] + "(" + safety[c_n_l[i]] + ")" + "avg:" + str(avg))
    
    plt.xlabel('Example Index', fontsize = 18)
    plt.ylabel('Negative curvature edges ratio', fontsize = 18)
    plt.legend(fontsize = 12, loc = 'best')
    plt.grid(True)
    
    plt.savefig(res_path + metric + "_" + s + ".png")
    plt.close()
    

    

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

    for t in types:
        for s in szs:
            frac= []
            c_n_l = []
            for c in cs:
                name = '_'.join([t,s,c])
                frac_name = metric + "_frac.pkl"
                frac_l = get_fraction(frac_name, name)
                frac.append(frac_l)
                c_n_l.append(name)
                
                print(f'The average ratio for controller {name} is {np.mean(frac_l):.5f}...')
                
            if (len(frac) > 0):
                draw(frac, c_n_l, t+s)
                
            print(f'Finish controller with size {s}...')

