import pickle
import numpy as np
import networkx as nx

import os
from collections import defaultdict, Counter

import sys
sys.path.append("..")

import tools.utils as utils
from tools.get_community import *

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")




def get_c_fc(curvature, b, prefix_dims, threshold = -50):
    c = []
    neg_e = set()   # Negative edges 
    pos_e = set()
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue
            c.append((i,j,curr))

    c.sort(key=lambda x: x[2])

    for (i,j,curr) in c:
        i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        i1 = (int)(i)
        j1 = (int)(j)
        if curr < 0:
            neg_e.add((i1,j1,curr))
        elif curr >= 0:
            pos_e.add((i1,j1,curr))

    return neg_e, pos_e



def get_c_cnn(curvature, b, prefix_dims, threshold = -50):
    c = []
    neg_e = set()  # Negative curvature edges
    pos_e = set()  # Positive curvature edges
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue

            c.append((i,j,curr))

    c.sort(key=lambda x: x[2])
    
    for (i,j,curr) in c:
        i1 = (int)(i)
        j1 = (int)(j)
        i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        if curr < 0 and i_layer >= 2:
            neg_e.add((i1,j1,curr))
        elif curr >= 0 and i_layer >= 2:
            pos_e.add((i1,j1,curr))
        
    return neg_e, pos_e



def cal_indegree_histogram(args):
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    data_path = args.mnist_data_path
    metric = args.metric
    dataset = args.dataset
    cifar = args.cifar
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    layer = [2]
    if model_type.lower() == "fc" and "big" not in model_pre_name.lower():
        layer = [2,4]
        
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    threshold_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    

    # Plot the data
    for layer_num in layer:
        if model_type.lower() == "fc":
            correct_suffix_res = dataset + "_res_correct.pkl"
            misclassified_suffix_res = dataset + "_res_misclassified.pkl"
            res_name = model_full_n + metric + '_' + str(layer_num) + correct_suffix_res
            misres_name = model_full_n + metric + '_' + str(layer_num) + misclassified_suffix_res
        # cnn model
        elif model_type.lower() == "cnn" and dataset.lower() == "mnist":
            correct_suffix_res = dataset + "_res_correct.pkl"
            misclassified_suffix_res = dataset + "_res_misclassified.pkl"    
            res_name = model_full_n + metric + '_' + correct_suffix_res
            misres_name = model_full_n + metric + '_' + misclassified_suffix_res    
            
        elif model_type.lower() == "cnn" and dataset.lower() == "cifar":
            robust_suffix_gs = "_graphsize_robust_cifar.pkl"
            norobust_suffix_gs = "_graphsize_norobust_cifar.pkl"
            robust_suffix_res = "_res_robust_cifar.pkl"
            norobust_suffix_res = "_res_norobust_cifar.pkl"
            
            if cifar.lower() == 'big':
                robust_suffix = "frac_robust_cifar_big.pkl"
                norobust_suffix = "frac_norobust_cifar_big.pkl"
            
        else:
            raise Exception("Invalid model type, model type should be {fc, fc_linear, cnn}!")
        
        with open(data_path + res_name, 'rb') as file:
            res_dict = pickle.load(file)
        
        tmp = res_path
        for thre in threshold_ratio:
            res_path = tmp + str(thre) + '/'

            if not os.path.exists(res_path):
                os.makedirs(res_path)
            
            with open(res_path + "common_edge" + ".txt", "w+") as ff:
                for l in selected_classes:
                    print(f"Processing label {l}")
                    ff.write(f'For label {l}: \n')
                    # Accumulate per-example degrees
                    all_gni_sets = []
                    prefix_dims = None
                    dim = None  # Will be consistent across examples
                    idx = 0

                    for (ricci, batch, dim, node_ori) in res_dict[l]:
                        prefix_dims = np.cumsum([0] + dim).tolist()
                        # neg_e, _ = get_c_fc(ricci, batch, prefix_dims)
                        
                        if model_type.lower() == "fc":
                            neg_e, _ = get_c_fc(ricci, batch, prefix_dims)
                        elif model_type.lower() == "cnn":
                            neg_e, _ = get_c_cnn(ricci, batch, prefix_dims)
                        else:
                            raise Exception("Invalid model type, model type should be {fc, cnn}!")

                        # Discard weights for counting only structure
                        neg_edges = set((i, j) for (i, j, _) in neg_e)
                        all_gni_sets.append(neg_edges)
                        
                        idx += 1
                        if idx >= 10:
                            break


                # Step 2: Take intersection (common edges across all examples)
                edge_counter = Counter()
                for gni in all_gni_sets:
                    edge_counter.update(gni)
                    
                total_graphs = len(all_gni_sets)
                low_freq_threshold = int(thre * total_graphs)  
                high_freq_threshold = int(0.9 * total_graphs)

                gn_edges = set(
                    edge for edge, count in edge_counter.items()
                    if low_freq_threshold <= count <= high_freq_threshold
                )
                
                print(f"Final Gn has {len(gn_edges)} edges (threshold = {thre})")
                ff.write(f"\nFinal Gn has {len(gn_edges)} edges (threshold = {thre})\n")
                
                # Step 3: Compute in-degrees for Gn
                in_degree_dict = defaultdict(int)
                for (src, tgt) in gn_edges:
                    in_degree_dict[tgt] += 1

                prefix_dims = np.cumsum([0] + dim).tolist()
                non_input_node_indices = [n for lidx in range(1, len(dim)) for n in range(prefix_dims[lidx], prefix_dims[lidx+1])]

                in_degrees = np.array([in_degree_dict.get(node, 0) for node in non_input_node_indices], dtype=np.int32)

                # Step 4: Save and Plot
                os.makedirs(res_path, exist_ok=True)
                out_file = os.path.join(res_path, f"label_{l}_final_gn_indegree.npz")
                np.savez(out_file,
                        in_degrees=in_degrees,
                        all_nodes=np.array(non_input_node_indices, dtype=np.int32),
                        label=l,
                        threshold_ratio=thre)