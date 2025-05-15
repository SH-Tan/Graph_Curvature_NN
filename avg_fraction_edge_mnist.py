import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict
import os

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


def get_fraction(curvature, b, dims):
    c = []
    layer_num = len(dims) - 1
    neg = np.zeros((layer_num), dtype=np.float32)
    top_neg = np.zeros((layer_num), dtype=np.float32)
    total_e = np.zeros((layer_num), dtype=np.float32)
    c_per_l = [[0],[0],[0],[0],[0],[0],[0]]
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
            
            if l >= layer_num-3:
                c.append(curr)
            c_per_l[l].append(curr)
    return neg, total_e, top_neg, c, c_per_l



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
    if model_type.lower() == "fc" and "big" not in model_pre_name.lower():
        layer = [2,4]
        
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    eps = [0.03, 0.07, 0.1, 0.2]
    Q = [1]
    # eps = [0.1]
    
    if dataset.lower() == "cifar":
        eps = [1,2,3,5]
                
    robust_suffix = "frac_robust.pkl"
    norobust_suffix = "frac_norobust.pkl"

    with open(res_path + "net_anysis2.txt", "w+") as ff:
        # Plot the data
        for layer_num in layer:
            ff.write(f'W = {metric}: For model {model_type} - {model_pre_name}, layer {layer_num}: \n')
            
            if model_type.lower() == "fc":
                robust_suffix_gs = dataset + "_graphsize_robust.pkl"
                norobust_suffix_gs = dataset + "_graphsize_norobust.pkl"
                robust_suffix_res = dataset + "_res_robust.pkl"
                norobust_suffix_res = dataset + "_res_norobust.pkl"
            
            # fc linear model
            elif model_type.lower() == "fc_linear":
                robust_suffix = "frac_robust_linear.pkl"
                norobust_suffix = "frac_norobust_linear.pkl"
                
            # cnn model
            elif model_type.lower() == "cnn" and dataset.lower() == "mnist":
                robust_suffix_gs = "_graphsize_robust_cnn.pkl"
                norobust_suffix_gs = "_graphsize_norobust_cnn.pkl"
                robust_suffix_res = "_res_robust_cnn.pkl"
                norobust_suffix_res = "_res_norobust_cnn.pkl"
                
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
    
            for q in Q:
                robust_area = defaultdict(list)
                norobust_area = defaultdict(list)

                for e in eps:
                    ff.write(f'For eps = {e}\n')
                    gs_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + robust_suffix_gs
                    nogs_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + norobust_suffix_gs
                    res_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + robust_suffix_res
                    nores_name = model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + norobust_suffix_res
           
                    if model_type.lower() == "cnn":
                        gs_name = model_full_n + str(e) + metric + str(q) + robust_suffix_gs
                        nogs_name = model_full_n + str(e) + metric + str(q) + norobust_suffix_gs
                        res_name = model_full_n + str(e) + metric + str(q) + robust_suffix_res
                        nores_name = model_full_n + str(e) + metric + str(q) + norobust_suffix_res
                    
                    
                    ff.write(f'For the robust examples: \n')
                    with open(data_path + gs_name, 'rb') as file:
                        gs_dict = pickle.load(file)
                        
                    with open(data_path + res_name, 'rb') as file:
                        res_dict = pickle.load(file)

                    count = 0
                    gs1 = 0.
                    gs2 = 0.
                    gs3 = 0.
                    for l in selected_classes:
                        # ff.write(f'For label {l}:\n')
                        for (sz1, sz2, sz3) in gs_dict[l]:
                            gs1 += sz1
                            gs2 += sz2
                            gs3 += sz3
                            count += 1
                        
                    gs1 = gs1/count if count > 0 else 0.
                    gs2 = gs2/count if count > 0 else 0.
                    gs3 = gs3/count if count > 0 else 0.
                    
                    if (gs1 > 0):
                        ff.write(f'The average graph is before normalization zero and nonzero edges are {gs2} and {gs1}, after normalization non-zero edges are {gs3}\n')
                        ff.write(f'remove edges {gs1-gs3}, remove edge ratio is {((gs1-gs3)/gs1) :.5f}...\n\n')
                    else:
                        ff.write(f'The average graph is before / after normalization is {gs1} and {gs2}, remove edges {gs1-gs2}\n\n')
                    
                    
                    count = 0
                    neg_c = 0.
                    topneg_c = 0.
                    total_e = 0.
                    frac_neg = 0.
                    frac_top = 0.
                    avg_c = 0.
                    med_c = 0.
                    low_c = []
                    high_c = 0.
                    var_c = 0.
                    c_neg = 0.
                    c_zero = 0.
                    c_pos = 0.
                    c_neg_f = 0.
                    c_zero_f = 0.
                    c_pos_f = 0.
                    total_c = 0.
                    last_5per_c = 0.
                    total_top_c = 0.
                    min_layer = 0.
                    for l in selected_classes:
                        # ff.write(f'For label {l}:\n')
                        for (ricci, batch, dim) in res_dict[l]:
                            neg, total, top, curv, curv_per_l = get_fraction(ricci, batch, dim)
                            c_num = len(curv)
                            curv = np.array(curv)
                            # if (np.min(curv) < -10000):
                            #     continue
                            
                            neg_c += neg
                            topneg_c += top
                            total_e += total
                            frac_neg += (neg/total)
                            frac_top += (top/total)
                            
                            min_values = [np.min(sublist) for sublist in curv_per_l if sublist]      
                            min_layer += np.array(min_values)
                            
                            total_c += c_num
                            total_top_c += (len(curv[curv < -10])/c_num)
                            c_neg += len(curv[curv<0])
                            c_zero += len(curv[curv==0])
                            c_pos += len(curv[curv>0])
                            c_neg_f += (len(curv[curv<0])/c_num)
                            c_zero_f += (len(curv[curv==0])/c_num)
                            c_pos_f += (len(curv[curv>0])/c_num)
                            last_5per_c += curv[-(int)(0.04*c_num)]
                            avg_c += np.mean(curv)
                            med_c += np.median(curv)
                            low_c.append(np.min(curv))
                            high_c += np.max(curv)
                            var_c += np.var(curv)
                            count += 1
                            
                    #         if (np.min(curv) != min_values[1]):
                    #             ff.write(f'{min_values} - {np.min(curv)}\n')
                            
                    # ff.write(f'\n\n')
                            
                    neg_c = neg_c/count if count > 0 else 0.
                    topneg_c = topneg_c/count if count > 0 else 0.
                    total_e = total_e/count if count > 0 else 0.
                    frac_neg = frac_neg/count if count > 0 else 0.
                    frac_top = frac_top/count if count > 0 else 0.
                    min_layer = min_layer/count if count > 0 else 0.

                    total_c = total_c/count if count > 0 else 0.
                    c_pos = c_pos/count if count > 0 else 0.
                    c_zero = c_zero/count if count > 0 else 0.
                    c_neg = c_neg/count if count > 0 else 0.
                    c_pos_f = c_pos_f/count if count > 0 else 0.
                    c_zero_f = c_zero_f/count if count > 0 else 0.
                    c_neg_f = c_neg_f/count if count > 0 else 0.
                    
                    last_5per_c = last_5per_c/count if count > 0 else 0.
                    total_top_c = total_top_c/count if count > 0 else 0.
                    
                    ff.write(f'For each layer the average total edge is {total_e}, the average negavtive curvature edge is {neg_c}, the average fraction is {frac_neg}...\n')
                    ff.write(f'For each layer the average top-k (< -10) negative curvature edge is {topneg_c}, the average fraction is {frac_top}, total average fraction for curv < -10 is {total_top_c:.3f}...\n')
                    if total_c > 0:
                        ff.write(f'The average total curvatrue number is {total_c}, postive is {c_pos} ({c_pos_f:.4f}), zero is {c_zero} ({c_zero_f:.4f}), negative is {c_neg} ({c_neg_f:.4f}). The last 4% curvature is {last_5per_c}\n\n')
                        ff.write(f'The average min value of each layer is {min_layer}\n\n')
                    # ff.write(f'For last layer the average total edge is {total_e_last}, the average negavtive curvature edge {neg_c_last}, the average top-k (<-10) negavtive curvature edge is {topneg_c_last}...\n')
                    # ff.write(f'For last layer, the average negavtive curvature edge fraction {frac_neg_last}, the average top-k (<-10) negavtive curvature edge fraction {frac_top_last}...\n\n')
                    avg_c = avg_c/count if count > 0 else 0.
                    med_c = med_c/count if count > 0 else 0.
                    low_c = np.median(low_c)
                    high_c = high_c/count if count > 0 else 0.
                    # last_10_percent_values = last_10_percent_values/count if count > 0 else 0.
                    var_c = var_c/count if count > 0 else 0.
                    
                    ff.write(f'For all the curvatures: \n')
                    ff.write(f'The average is {avg_c :.3f}, lowest is {low_c :.3f}, the highest is {high_c :.3f}, the median is {med_c :.3f}, variance is {var_c :.3f}\n\n')

                    
                    ff.write(f'For the non-robust examples: \n')
                    with open(data_path + nogs_name, 'rb') as file:
                        nogs_dict = pickle.load(file)
                        
                    with open(data_path + nores_name, 'rb') as file:
                        nores_dict = pickle.load(file)
                    
                    count = 0
                    gs1 = 0.
                    gs2 = 0.
                    gs3 = 0.
                    for l in selected_classes:
                        # ff.write(f'For label {l}:\n')
                        for (sz1, sz2, sz3) in nogs_dict[l]:
                            gs1 += sz1
                            gs2 += sz2
                            gs3 += sz3
                            count += 1
                        
                    gs1 = gs1/count if count > 0 else 0.
                    gs2 = gs2/count if count > 0 else 0.
                    gs3 = gs3/count if count > 0 else 0.
                    
                    if (gs1 > 0):
                        ff.write(f'The average graph is before normalization zero and nonzero edges are {gs2} and {gs1}, after normalization non-zero edges are {gs3}\n')
                        ff.write(f'remove edges {gs1-gs3}, remove edge ratio is {((gs1-gs3)/gs1) :.5f}...\n\n')
                    else:
                        ff.write(f'The average graph is before / after normalization is {gs1} and {gs2}, remove edges {gs1-gs2}\n\n')
                    
                    
                    count = 0
                    neg_c = 0.
                    topneg_c = 0.
                    total_e = 0.
                    frac_neg = 0.
                    frac_top = 0.
                    avg_c = 0.
                    med_c = 0.
                    low_c = []
                    high_c = 0.
                    var_c = 0.
                    c_neg = 0.
                    c_zero = 0.
                    c_pos = 0.
                    c_neg_f = 0.
                    c_zero_f = 0.
                    c_pos_f = 0.
                    total_c = 0.
                    last_5per_c = 0.
                    total_top_c = 0.
                    min_layer = 0.
                    for l in selected_classes:
                        # ff.write(f'For label {l}:\n')
                        for (ricci, batch, dim) in nores_dict[l]:
                            neg, total, top, curv, curv_per_l = get_fraction(ricci, batch, dim)
                            c_num = len(curv)
                            curv = np.array(curv)
                            # if (np.min(curv) < -10000):
                            #     continue
                            
                            neg_c += neg
                            topneg_c += top
                            total_e += total
                            frac_neg += (neg/total)
                            frac_top += (top/total)
                            
                            min_values = [min(sublist) for sublist in curv_per_l if sublist]
                            min_layer += np.array(min_values)
                            
                            total_c += c_num
                            total_top_c += (len(curv[curv < -10])/c_num)
                            c_neg += len(curv[curv<0])
                            c_zero += len(curv[curv==0])
                            c_pos += len(curv[curv>0])
                            c_neg_f += (len(curv[curv<0])/c_num)
                            c_zero_f += (len(curv[curv==0])/c_num)
                            c_pos_f += (len(curv[curv>0])/c_num)
                            last_5per_c += curv[-(int)(0.04*c_num)]
                            avg_c += np.mean(curv)
                            med_c += np.median(curv)
                            low_c.append(np.min(curv))
                            high_c += np.max(curv)
                            var_c += np.var(curv)
                            count += 1
        
                    neg_c = neg_c/count if count > 0 else 0.
                    topneg_c = topneg_c/count if count > 0 else 0.
                    total_e = total_e/count if count > 0 else 0.
                    frac_neg = frac_neg/count if count > 0 else 0.
                    frac_top = frac_top/count if count > 0 else 0.
                    min_layer = min_layer/count if count > 0 else 0.

                    total_c = total_c/count if count > 0 else 0.
                    c_pos = c_pos/count if count > 0 else 0.
                    c_zero = c_zero/count if count > 0 else 0.
                    c_neg = c_neg/count if count > 0 else 0.
                    c_pos_f = c_pos_f/count if count > 0 else 0.
                    c_zero_f = c_zero_f/count if count > 0 else 0.
                    c_neg_f = c_neg_f/count if count > 0 else 0.
                    
                    last_5per_c = last_5per_c/count if count > 0 else 0.
                    total_top_c = total_top_c/count if count > 0 else 0.
                    
                    ff.write(f'For each layer the average total edge is {total_e}, the average negavtive curvature edge is {neg_c}, the average fraction is {frac_neg}...\n')
                    ff.write(f'For each layer the average top-k (< -10) negative curvature edge is {topneg_c}, the average fraction is {frac_top}, total average fraction for curv < -10 is {total_top_c:.3f}...\n')
                    if total_c > 0:
                        ff.write(f'The average total curvatrue number is {total_c}, postive is {c_pos} ({c_pos_f:.4f}), zero is {c_zero} ({c_zero_f:.4f}), negative is {c_neg} ({c_neg_f:.4f}). The last 4% curvature is {last_5per_c}\n\n')
                        ff.write(f'The average min value of each layer is {min_layer}\n\n')
                    # ff.write(f'For last layer the average total edge is {total_e_last}, the average negavtive curvature edge {neg_c_last}, the average top-k (<-10) negavtive curvature edge is {topneg_c_last}...\n')
                    # ff.write(f'For last layer, the average negavtive curvature edge fraction {frac_neg_last}, the average top-k (<-10) negavtive curvature edge fraction {frac_top_last}...\n\n')

                    avg_c = avg_c/count if count > 0 else 0.
                    med_c = med_c/count if count > 0 else 0.
                    low_c = np.median(low_c)
                    high_c = high_c/count if count > 0 else 0.
                    # last_10_percent_values = last_10_percent_values/count if count > 0 else 0.
                    var_c = var_c/count if count > 0 else 0.
                    
                    ff.write(f'For all the curvatures: \n')
                    ff.write(f'The average is {avg_c :.3f}, lowest is {low_c :.3f}, the highest is {high_c :.3f}, the median is {med_c :.3f}, variance is {var_c :.3f}\n\n')

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