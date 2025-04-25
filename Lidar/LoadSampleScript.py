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
from collections import defaultdict
from Lidar.lidar_tools.getNNweights import getNN_info
from Lidar.lidar_tools.graph_cal import *
from Lidar.lidar_tools.graph_curvature_multihops import graph_curvature_main_torch

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



# def get_fraction(curvature, b):
#     c = []
#     neg = []
#     total_e = []
#     for i in range(b):
#         curr = np.array(curvature[i])
#         neg.append(len(curr[curr<0]))
#         total_e.append(len(curr))
#     return np.array(neg), np.array(total_e), curr


def get_fraction(curvature, b, dims):
    c = []
    layer_num = len(dims) - 1
    neg = np.zeros((layer_num), dtype=np.float32)
    top_neg = np.zeros((layer_num), dtype=np.float32)
    total_e = np.zeros((layer_num), dtype=np.float32)
    # neg = 0.
    # total_e = 0.
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue
    
            l = int(i)
            if curr < 0:
                neg[l] += 1
            if curr < -10:
                top_neg[l] += 1
            total_e[l] += 1
            c.append(curr)
    return neg, total_e, top_neg, c



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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    controllers = load_all_controllers()
    
    res_path = args.lidar_res_path
    metric = args.metric
    alpha = args.alpha
    hops = args.hops
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    
    with open("Lidar/synthetic.yml", 'r') as f:
        lidar_traj_by_controller = yaml.full_load(f)
    
    keys = lidar_traj_by_controller.keys()
    print("Keys = ", keys)
  
  
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64','128']
    
    num_samples = 80

    for t in types:
        for s in szs:
            for c in cs:
                gs_l = defaultdict(list) # graph size
                en_l = defaultdict(list) # edge num
                el_l = defaultdict(list) # per layer
                curv_l = defaultdict(list) # curvature
                res_l = defaultdict(list)
    
                name = '_'.join([t,s,c])
                cur_c = controllers[name]
                
                print(f'Controller: {name}')
                
                traj_l = lidar_traj_by_controller[name]

                # all the trajectories for one controller
                for i in range(len(traj_l)):
                    traj = np.array(traj_l[i]) # 219 * 21
                    sampled_inputs = traj[np.random.randint(traj.shape[0], size=num_samples), :] # sample * 21
                     
                    graph_size = []
                    edge_n = []
                    edge_layer = []
                    curv_n = []
                    res = []
                    
                    for index, input in enumerate(sampled_inputs):
                        input = np.array(input)
                        input = input.reshape(1, -1)
                        input = torch.tensor(input)
                        dims, nodes_num, edges_num, NN_w, edge_v, nodes = getNN_info(cur_c, input, device)
                        
                        if metric.lower() == "q_ngr":
                            weights = NN_w.detach().clone().to(device)                   
                            weights[edge_v == 0] = 0.
                        
                        elif metric.lower() == "q_inv":
                            weights = NN_w.detach().clone().to(device)                   
                            weights[edge_v == 0] = 0.
                            
                        elif metric.lower() == "q_exp":
                            weights = edge_v.detach().clone().to(device) 
                            
                        if metric.lower() == "q_ngr":
                            weights_inv, weights_inv2 = normalization_weight_w1(nodes, weights, dims)
                            weights_inv = weights_inv.detach()
                            weights_inv2 = weights_inv2.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, alpha=alpha, hops=hops)
                            
                        elif metric.lower() == "q_inv":
                            weights_inv = normalization_weight_w2(nodes, weights, dims)
                            weights_inv = weights_inv.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, alpha=alpha, hops=hops)
            
                        elif metric.lower() == "q_exp":
                            weights_inv = normalization_weight_w6(nodes, weights, dims, q=1)
                            weights_inv = weights_inv.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, alpha=alpha)
                        else:
                            raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")

                        # neg_num, total_edge, top_neg_num, curv = get_fraction(ricci_curvature, weights_inv.shape[0], dims)

                        # graph size before/after normalization
                        graph_size.append((len(weights[weights!=0]),len(weights[weights==0]),len(weights_inv[weights_inv!=0])))
                        # # edge num
                        # edge_n.append((len(weights[weights==0]), len(weights[weights!=0])))
                        # # per layer
                        # edge_layer.append((neg_num, top_neg_num, total_edge))
                        # # curvature
                        # curv_n.append(curv)
                        # all the results
                        res.append((ricci_curvature, weights_inv.shape[0], dims))
                        
                 
                    print(f'Finish {i} trajectory...')
                  
                    gs_l[i].append(graph_size)
                    # en_l[i].append(edge_n)
                    # el_l[i].append(edge_layer)
                    # curv_l[i].append(curv_n)
                    res_l[i].append(res)
                
                
                with open(res_path + metric + name + "_res.pkl", 'wb') as file:
                    pickle.dump(res_l, file)
                    
                with open(res_path + metric + name + "_graphsize.pkl", 'wb') as file:
                    pickle.dump(gs_l, file)

                # with open(res_path + metric + name + "_edgenum.pkl", 'wb') as file:
                #     pickle.dump(en_l, file)
                
                # with open(res_path + metric + name + "_perlayer.pkl", 'wb') as file:
                #     pickle.dump(el_l, file)
                    
                # with open(res_path + metric + name + "_curv.pkl", 'wb') as file:
                #     pickle.dump(curv_l, file)
 
 
        
