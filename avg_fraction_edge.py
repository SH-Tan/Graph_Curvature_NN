import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict
import os

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



def avg_f_cal(args):
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    data_path = args.mnist_data_path
    metric = args.metric
    dataset = args.dataset
    cifar = args.cifar
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    layer = [2]
    if model_type.lower() == "fc" and model_pre_name.lower() != 'big':
        layer = [2,4]
        
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    eps = [0.03, 0.07, 0.1, 0.2]
    Q = [1]
    # eps = [0.1]
    
    if dataset.lower() == "cifar":
        eps = [1,2,3,5]
                
    robust_suffix = "frac_robust.pkl"
    norobust_suffix = "frac_norobust.pkl"

    with open(res_path + "avg_f.txt", "a+") as ff:
        # Plot the data
        for layer_num in layer:
            ff.write(f'W = {metric}: For model {model_type} - {model_pre_name}, layer {layer_num}: \n')
            
            if model_type.lower() == "fc":
                robust_suffix = dataset + "frac_robust.pkl"
                norobust_suffix = dataset + "frac_norobust.pkl"
                robust_suffix_curv = dataset + "curv_robust.pkl"
                norobust_suffix_curv = dataset + "curv_norobust.pkl"
                robust_suffix_edge = dataset + "edge_robust.pkl"
                norobust_suffix_edge = dataset + "edge_norobust.pkl"
            
            # fc linear model
            elif model_type.lower() == "fc_linear":
                robust_suffix = "frac_robust_linear.pkl"
                norobust_suffix = "frac_norobust_linear.pkl"
                
            # cnn model
            elif model_type.lower() == "cnn" and dataset.lower() == "mnist":
                robust_suffix = "frac_robust_cnn.pkl"
                norobust_suffix = "frac_norobust_cnn.pkl"
                robust_suffix_curv = "curv_robust_cnn.pkl"
                norobust_suffix_curv = "curv_norobust_cnn.pkl"
                robust_suffix_edge = "edge_robust_cnn.pkl"
                norobust_suffix_edge = "edge_norobust_cnn.pkl"
                
            elif model_type.lower() == "cnn" and dataset.lower() == "cifar":
                robust_suffix = "frac_robust_cifar.pkl"
                norobust_suffix = "frac_norobust_cifar.pkl"
                robust_suffix_curv = "curv_robust_cifar.pkl"
                norobust_suffix_curv = "curv_norobust_cifar.pkl"
                robust_suffix_edge = "edge_robust_cifar.pkl"
                norobust_suffix_edge = "edge_norobust_cifar.pkl"
                
                if cifar.lower() == 'big':
                    robust_suffix = "frac_robust_cifar_big.pkl"
                    norobust_suffix = "frac_norobust_cifar_big.pkl"
                
            else:
                raise Exception("Invalid model type, model type should be {fc, fc_linear, cnn}!")
    
            for q in Q:
                robust_area = defaultdict(list)
                norobust_area = defaultdict(list)

                for e in eps:
                    frac_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + robust_suffix
                    nofrac_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + norobust_suffix
                    curv_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + robust_suffix_curv
                    nocurv_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + norobust_suffix_curv
                    edge_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + robust_suffix_edge
                    noedge_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + norobust_suffix_edge
                    
                    if model_type.lower() == "cnn":
                        frac_name = model_full_n + str(e) + metric + str(q) + robust_suffix
                        nofrac_name = model_full_n + str(e) + metric + str(q) + norobust_suffix
                        curv_name = model_full_n + str(e) + metric + str(q) + robust_suffix_curv
                        nocurv_name = model_full_n + str(e) + metric + str(q) + norobust_suffix_curv
                        edge_name = model_full_n + str(e) + metric + str(q) + robust_suffix_edge
                        noedge_name = model_full_n + str(e) + metric + str(q) + norobust_suffix_edge
                    
                    
                    with open(data_path + frac_name, 'rb') as file:
                        neg_dict = pickle.load(file)
                        
                    with open(data_path + curv_name, 'rb') as file:
                        total_dict = pickle.load(file)
                        
                    with open(data_path + edge_name, 'rb') as file:
                        edge_dict = pickle.load(file)
                    
                    neg = 0.
                    total = 0.
                    count = 0
                    frac = 0.
                    ori_edge = 0.
                    norm_edge1 = 0.
                    norm_edge2 = 0.
                    remove_edge = 0.
                    for l in selected_classes:
                        for ((l_w, l_w_0), big_w) in zip(total_dict[l], neg_dict[l]):
                            neg += l_w
                            total += l_w_0
                            frac += big_w
                            count += 1
                            
                        print(f'For {l}, eps = {e}, {count}')
                        
                    neg = neg/count if count > 0 else 0.
                    total = total/count if count > 0 else 0.
                    frac = frac/count if count > 0 else 0.
                    
                    # ff.write(f'For eps = {e} of robust images, for each layer the average total edge is {total}, the average negavtive curvature edge is {neg}, the average fraction is {frac}...\n')
                    ff.write(f'For eps = {e} of robust images, have total weights {neg}, have 0 input is {total}, the average 0 input have big weights is {frac}...\n')

                    count = 0
                    for l in selected_classes:
                        for (e1,e2,e3) in edge_dict[l]:
                            ori_edge += e1
                            norm_edge1 += e2
                            norm_edge2 += e3
                            count += 1
                            remove_edge += (e1-e2)/e1
                        
                    ori_edge = ori_edge/count if count > 0 else 0.
                    norm_edge1 = norm_edge1/count if count > 0 else 0.
                    norm_edge2 = norm_edge2/count if count > 0 else 0.
                    remove_edge = remove_edge/count if count > 0 else 0.
                    
                    ff.write(f'The average total edge before normalization is {ori_edge}, after normalization is {norm_edge1} and {norm_edge2}, removed edge ratio is {remove_edge}...\n\n')
                    
                    
                    
                    
                    with open(data_path + nofrac_name, 'rb') as file:
                        neg_dict = pickle.load(file)
                        
                    with open(data_path + nocurv_name, 'rb') as file:
                        total_dict = pickle.load(file)
                        
                        
                    with open(data_path + noedge_name, 'rb') as file:
                        edge_dict = pickle.load(file)
                    
                    neg = 0.
                    total = 0.
                    count = 0
                    frac = 0.
                    ori_edge = 0.
                    norm_edge1 = 0.
                    norm_edge2 = 0.
                    remove_edge = 0.
                    
                    for l in selected_classes:
                        for ((l_w, l_w_0), big_w) in zip(total_dict[l], neg_dict[l]):
                            neg += l_w
                            total += l_w_0
                            frac += big_w
                            count += 1
                            
                        print(f'For {l}, eps = {e}, {count}')
                        
                    neg = neg/count if count > 0 else 0.
                    total = total/count if count > 0 else 0.
                    frac = frac/count if count > 0 else 0.
                        
                    # ff.write(f'For eps = {e} of non-robust images, for each layer the average total edge is {total}, the average negavtive curvature edge is {neg}, the average fraction is {frac}...\n')
                    ff.write(f'For eps = {e} of robust images, have total weights {neg}, have 0 input is {total}, the average 0 input have big weights is {frac}...\n')
                    
                    count = 0
                    for l in selected_classes:
                        for (e1,e2,e3) in edge_dict[l]:
                            ori_edge += e1
                            norm_edge1 += e2
                            norm_edge2 += e3
                            count += 1
                            remove_edge += (e1-e2)/e1
                        
                    ori_edge = ori_edge/count if count > 0 else 0.
                    norm_edge1 = norm_edge1/count if count > 0 else 0.
                    norm_edge2 = norm_edge2/count if count > 0 else 0.
                    remove_edge = remove_edge/count if count > 0 else 0.
                    
                    ff.write(f'The average total edge before normalization is {ori_edge}, after normalization is {norm_edge1} and {norm_edge2}, removed edge ratio is {remove_edge}...\n\n')
                    
                # markers = ['.', 'o', '*', '+', '>', 'v']

                # i = 0
                
                # fig1, ax1 = plt.subplots(figsize=(9, 7))

                # for e in eps:
                #     ax1.plot(range(len(robust_area[e])), robust_area[e], linewidth = 1.5, marker = markers[i], ms = 9, label = f"eps = {e}")
                #     i += 1
                    
                # plt.xlabel('Image label', fontsize = 19, fontweight='semibold')
                # plt.ylabel('Nagatiev Curvature Edge Ratio', fontsize = 19, fontweight='semibold')

                # x_ticks = np.arange(0, 10, 1)
                # plt.yticks(size=18,weight='semibold')
                # plt.xticks(size=18,weight='semibold')
                    
                # plt.legend(fontsize = 20, loc = 'best', prop={'size':19, 'weight':'semibold'})
                # plt.grid(True)
                # plt.savefig(res_path + model_n + str(e) + q_str + str(layer_num) + "_frac.eps")
                # plt.close()
                
                # for e in eps:
                    
                #     ax1.plot(range(len(robust_area[e])), robust_area[e], linewidth = 1.5, marker = markers[i], ms = 9, label = f"eps = {e} robust")
                #     ax1.plot(range(len(norobust_area[e])), norobust_area[e], linewidth = 1.5, marker = markers[i], ms = 9, label = f"eps = {e} nonrobust")
                #     i += 1
                    
                #     plt.title('Two-layer FC network: negative curvature edges fraction', fontsize=24)
                #     plt.xlabel('Image label', fontsize = 25)
                #     plt.ylabel('Nagatiev Curvature Edge Ratio', fontsize = 25)

                #     x_ticks = np.arange(0, 10, 1)
                #     plt.xticks(x_ticks, fontsize = 20)

                #     plt.yticks(size = 20)
                        
                #     plt.legend(fontsize = 20, loc = 'best')
                #     plt.grid(True)
                #     plt.savefig(res_path + model_n + str(e) + q_str + str(q) + "_frac.png")
                #     plt.close()