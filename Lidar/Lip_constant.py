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


def load_all_controllers(controller_dir = './controllers/'):
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



def get_layer_lipschitz(w):
    weight = w.reshape(w.shape[0], -1) 
    weight = torch.from_numpy(weight)
    s = torch.linalg.svdvals(weight)
    return s.max().item()

    

def get_model_lipschitz_per_layer(model):
    lipschitz_constants = {}
    weights = {}
    offsets = {}

    layerCount = 1
    activations = []
    
    for layer in range(1, len(model['weights']) + 1):
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])
        
    lip = 1.
    
    for layer in range(layerCount-1):
        lipschitz = get_layer_lipschitz(weights[layer+1])
        if lipschitz is not None:
            lipschitz_constants[layer+1] = lipschitz
            lip *= lipschitz
    lipschitz_constants['global'] = lip
    return lipschitz_constants




if __name__=='__main__':

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
    
    # with open("./synthetic.yml", 'r') as f:
    #     lidar_traj_by_controller = yaml.full_load(f)
    
    # keys = lidar_traj_by_controller.keys()
    # print("Keys = ", keys)
  
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

                lipschitz_per_layer = get_model_lipschitz_per_layer(cur_c)
                for name, lip in lipschitz_per_layer.items():
                    print(f"Lipschitz constant of layer '{name}': {lip:.4f}")