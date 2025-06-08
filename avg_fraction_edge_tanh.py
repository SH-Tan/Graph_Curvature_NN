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
    c_first_layer = []
    layer_num = len(dims) - 1
    neg = np.zeros((layer_num), dtype=np.float32)
    zero_e = np.zeros((layer_num), dtype=np.float32)
    total_e = np.zeros((layer_num), dtype=np.float32)
    c_per_l = [[0],[0],[0],[0],[0],[0],[0]]
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue
            
            i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
            
            l = int(i_layer)
            if curr < 0:
                neg[l] += 1
            elif curr == 0.:
                zero_e[l] += 1
            total_e[l] += 1
            
            if curr < 0 and l == 0:
                c_first_layer.append(curr)
            elif curr < 0 and l > 0:
                c.append(curr)
            c_per_l[l].append(curr)
    return neg, total_e, c, c_per_l, c_first_layer, zero_e



def avg_f_cal_tanh(args):
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

    with open(res_path + "net_anysis1.txt", "a+") as ff:
        # Plot the data
        for layer_num in layer:
            ff.write(f'\nW = {metric}: For model {model_type} - {model_pre_name}, layer {layer_num}: \n')
            
            if model_type.lower() == "fc":
                correct_suffix_res = dataset + "_res_correct.pkl"
                misclassified_suffix_res = dataset + "_res_misclassified.pkl"
            
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

            res_name = model_full_n + metric + '_' + str(layer_num) + correct_suffix_res
            misres_name = model_full_n + metric + '_' + str(layer_num) + misclassified_suffix_res

            with open(data_path + res_name, 'rb') as file:
                res_dict = pickle.load(file)

            count = 0
            neg_c = 0.
            zero_c = 0.
            pos_c = 0.
            topneg_c = 0.
            total_e = 0.
            frac_neg = 0.
            frac_top = 0.
            avg_c = 0.
            med_c = 0.
            low_c = []
            low_c_first_layer = []
            high_c = 0.
            var_c = 0.
            var_c_first = 0.
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
                    neg, total, curv, curv_per_l, c_first_layer, zero_e = get_fraction(ricci, batch, dim)
                    # c_num = len(curv)
                    curv = np.array(curv)
                    c_first_layer = np.array(c_first_layer)
                    
                    neg_c += neg
                    zero_c += zero_e
                    pos_c += (total-neg-zero_e)
                    # print(f'{total}- {neg} - {zero_e} - {pos_c}')
                    total_e += total
                    frac_neg += (neg/total)
                    
                    min_values = [np.min(sublist) for sublist in curv_per_l if sublist]      
                    min_layer += np.array(min_values)
                    
                    # total_c += c_num
                    # total_top_c += (len(curv[curv < -10])/c_num)
                    # c_neg += len(curv[curv<0])
                    # c_zero += len(curv[curv==0])
                    # c_pos += len(curv[curv>0])
                    # c_neg_f += (len(curv[curv<0])/c_num)
                    # c_zero_f += (len(curv[curv==0])/c_num)
                    # c_pos_f += (len(curv[curv>0])/c_num)
                    # last_5per_c += curv[-(int)(0.04*c_num)]
                    avg_c += np.mean(curv)
                    med_c += np.median(curv)
                    low_c.append(np.min(curv))
                    low_c_first_layer.append(np.min(c_first_layer))
                    high_c += np.max(curv)
                    var_c += np.var(curv)
                    var_c_first += np.var(c_first_layer)
                    count += 1
                    
                    
            neg_c = neg_c/count if count > 0 else 0.
            zero_c = zero_c/count if count > 0 else 0.
            pos_c = pos_c/count if count > 0 else 0.
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
            
            ff.write(f'For each layer the average total edge number is {total_e}\n')
            ff.write(f"the average negavtive curvature edge is {neg_c}, the fraction is {neg_c/total_e}\n")
            ff.write(f'the average zero curvature edge is {zero_c}\n')
            ff.write(f'the average postive curvature edge is {pos_c}, the fraction is {pos_c/total_e}...\n')
            # ff.write(f'For each layer the average top-k (< -10) negative curvature edge is {topneg_c}, the average fraction is {frac_top}, total average fraction for curv < -10 is {total_top_c:.3f}...\n')
            # if total_c > 0:
            #     ff.write(f'The average total curvatrue number is {total_c}, postive is {c_pos} ({c_pos_f:.4f}), zero is {c_zero} ({c_zero_f:.4f}), negative is {c_neg} ({c_neg_f:.4f}). The last 4% curvature is {last_5per_c}\n\n')
            #     ff.write(f'The average min value of each layer is {min_layer}\n\n')
            # ff.write(f'For last layer the average total edge is {total_e_last}, the average negavtive curvature edge {neg_c_last}, the average top-k (<-10) negavtive curvature edge is {topneg_c_last}...\n')
            # ff.write(f'For last layer, the average negavtive curvature edge fraction {frac_neg_last}, the average top-k (<-10) negavtive curvature edge fraction {frac_top_last}...\n\n')
            avg_c = avg_c/count if count > 0 else 0.
            med_c = med_c/count if count > 0 else 0.
            high_c = high_c/count if count > 0 else 0.
            var_c = var_c/count if count > 0 else 0.
            var_c_first = var_c_first/count if count > 0 else 0.
            
            ff.write(f'\nFor all the curvatures: \n')
            ff.write(f'The first layer: average lowest is {np.mean(low_c_first_layer)} median lowest is {np.median(low_c_first_layer) :.3f}, variance is {var_c_first:.3f}\n')
            ff.write(f'The later layer: total average curvature is {avg_c :.3f}, average lowest is {np.mean(low_c)} median lowest is {np.median(low_c) :.3f}, the median is {med_c :.3f}, variance is {var_c :.3f}\n\n')

                    
                