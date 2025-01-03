import torch
import numpy as np
import random
import os
import torch.nn as nn
from collections import defaultdict

import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression as lg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

safety = {
    "DDPG_64_1": "0%",
    "DDPG_64_2": "20%",
    "DDPG_64_3": "80%",
    "DDPG_128_1": "80%",
    "DDPG_128_2": "40%",
    "DDPG_128_3": "0%",
    "TD3_64_1": "90%",
    "TD3_64_2": "90%",
    "TD3_64_3": "90%",
    "TD3_128_1": "90%",
    "TD3_128_2": "50%",
    "TD3_128_3": "90%"    
}


types = ['DDPG', 'TD3']
cs = ['1','2','3']
szs = ['64','128']

markers = ['*', '3', 'o', '+', '.', '>', 'v']
Keys =  ['DDPG_128_1', 'DDPG_128_2', 'DDPG_128_3', 'DDPG_64_1', 'DDPG_64_2', 'DDPG_64_3', 'TD3_128_1', 'TD3_128_2', 'TD3_128_3', 'TD3_64_1', 'TD3_64_2', 'TD3_64_3']


def get_fraction(name, data_path):
    with open(data_path + name, 'rb') as file:
        data = pickle.load(file)
        
    return data



# AUC, loss = zip(*sorted(zip(AUC, loss))) 
# legend: safety of controller
def draw(frac, c_n_l, s, res_path, metric, traj = ''): 
    assert(len(frac) == len(c_n_l))
    
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    y = [0.15, 0.19, 0.21, 0.25, 0.27, 0.31, 0.35, 0.39, 0.43, 0.45, 0.49, 0.51, 0.55]
    x = [1,2,3,4,5,6,7,8,9,10]
    
    for i in range(len(frac)):
        avg = round(np.mean(frac[i][0]),4)
        ax1.plot(range(len(frac[i])), frac[i], marker = markers[i], linewidth=2, markersize=15, label=c_n_l[i] + " (Safety: " + safety[c_n_l[i]] + ")")
    
    # avg = round(np.mean(frac['1'][0]),4)
    # ax1.plot(range(len(frac['1'][0])), frac['1'][0], marker = markers[1], label=c_n_l + "(" + safety[c_n_l] + ")" + "unsafe avg:" + str(avg))
    
    plt.title('Evaluation of [' + s + ',' + s + '] Controllers', fontsize = 23, fontweight='semibold')
    plt.xlabel('Trajectory Index', fontsize = 18, fontweight='semibold')
    plt.ylabel('Negative curvature edges fraction', fontsize = 19, fontweight='semibold')
    plt.yticks(y, size=18,weight='semibold')
    plt.xticks(x, size=18,weight='semibold')
    plt.legend(loc = 'best', prop={'size':13, 'weight':'semibold'})
    plt.grid(True)
    
    plt.savefig(res_path + metric + "_" + s + "_" + ".png")
    plt.close()

    

    

def lidar_draw(args):
    res_path = args.lidar_res_path
    data_path = args.lidar_data_path
    metric = args.metric
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
     
    for s in szs:   
        frac = []
        c_n_l = []  
        for t in types:
            for c in cs: 
                name = '_'.join([t,s,c])
                
                frac_name = metric + name + "_frac2.pkl"
                frac_l = get_fraction(frac_name, data_path)
                
                tmp = []
                for i in range(len(frac_l)):
                    tmp.append(np.mean(frac_l[i][0]))
                    
                frac.append(tmp)
                c_n_l.append(name)
                
                # print(f'The average ratio for controller {name} is {np.mean(frac_l):.5f}...')
        draw(frac, c_n_l, s, res_path, metric)

