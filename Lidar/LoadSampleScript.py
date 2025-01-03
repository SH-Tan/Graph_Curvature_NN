# load targets and input data

# compute control error on the training controllers and data
# compute control error on the holdout validation controllers
# to get confidence intervals


# to get confidence intervals for each of the rays, 
# compute outputs on validation set (sigmoids)

import yaml
import torch
import numpy as np
import os
import random
import pickle
from Lidar.getNNweights import getNN_info
from Lidar.graph_cal import *

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

Keys =  ['DDPG_128_1', 'DDPG_128_2', 'DDPG_128_3', 'DDPG_64_1', 'DDPG_64_2', 'DDPG_64_3', 'TD3_128_1', 'TD3_128_2', 'TD3_128_3', 'TD3_64_1', 'TD3_64_2', 'TD3_64_3']


def custom_predict_yaml(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount-1):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = 1/(1 + np.exp(-curNeurons))
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return np.array(curNeurons)

def vectorize_predict_yaml(model, inputs, temps=1):
    weights = {}
    offsets = {}

    layerCount = 1
    activations = []
    

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs.T
    
    print('Layer num: ', layerCount)

    for layer in range(layerCount-1):

        curNeurons = weights[layer+1]@curNeurons + offsets[layer+1].reshape(len(offsets[layer+1]),1)
        if 'Sigmoid' in activations[layer]:
            curNeurons = 1/(1 + np.exp(-curNeurons))
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return np.array(curNeurons).T


def load_all_controllers(controller_dir = 'Lidar/controllers/'):
    # load a dictionary of all other controller?
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64','128']
    controllers = {}
    for t in types:
        for s in szs:
            for c in cs:
                controllers_filename = controller_dir+t+'_L21_'+s+'x'+s+'_C'+c+'.yml'
                with open(controllers_filename, 'rb') as f:
                    controllers['_'.join([t,s,c])] = yaml.full_load(f)

    print("controller keys: ", list(controllers.keys()))
    return controllers



def start_lidar(args):
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    controllers = load_all_controllers()
    
    res_path = args.lidar_res_path
    metric = args.metric
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    
    with open("Lidar/lidar_trajectories_by_controller.yml", 'r') as f:
        lidar_traj_by_controller = yaml.full_load(f)
    
    keys = lidar_traj_by_controller.keys()
    print("Keys = ", keys)
  
  
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64', '128']
 
    
    for t in types:
        for s in szs:
            for c in cs:
                c_l = defaultdict(list)
                f_l = defaultdict(list)
    
                name = '_'.join([t,s,c])
                cur_c = controllers[name]
                
                print(f'Controller: {name}')
                
                traj_l = lidar_traj_by_controller[name]

                for i in range(len(traj_l)):
                    traj = traj_l[i]
                    tmp_f = []
                    tmp_c = []
                    
                    for index, input in enumerate(traj):
                        input = np.array(input)
                        input = input.reshape(1, -1)
                        input = torch.tensor(input)
                        dims, nodes_num, edges_num, NN_w, edge_v, nodes, nodes_ori = getNN_info(cur_c, input)
                        
                        # build graph
                        adj = build_adjm(nodes_num, dims, edge_v, nodes, NN_w)
                        adj = adj.cpu().detach().numpy()
                        curv, ratio, G = cal_curvature(adj, nodes, dims, nodes_ori.cpu())

                        tmp_f.append(ratio)
                        tmp_c.append(curv)
                  
                    c_l[i].append(tmp_c)
                    f_l[i].append(tmp_f)
                    
                    
                with open(res_path + metric + name + "_frac.pkl", 'wb') as file:
                    pickle.dump(f_l, file)

                with open(res_path + metric + name + "_curvature.pkl", 'wb') as file:
                    pickle.dump(c_l, file)
 
        
