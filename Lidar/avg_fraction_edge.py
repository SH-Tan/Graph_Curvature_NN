import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict
import os

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


def get_fraction(curvature, b, dims):
    c = []
    layer_num = len(dims) - 1
    neg = np.zeros((layer_num), dtype=np.float32)
    top_neg = np.zeros((layer_num), dtype=np.float32)
    total_e = np.zeros((layer_num), dtype=np.float32)
    c_per_l = [[],[],[]]
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
            if curr < -15:
                top_neg[l] += 1
            total_e[l] += 1
            c.append(curr)
            c_per_l[l].append(curr)
    return neg, total_e, top_neg, c, c_per_l



def avg_statics_lidar(args):
    res_path = args.lidar_res_path
    data_path = args.lidar_data_path
    metric = args.metric

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    with open(res_path + "avg_f_all.txt", "w+") as ff:
        for t in types:
            for s in szs:
                for c in cs:
                    name = '_'.join([t,s,c])
                    
                    print(f'Controll {name}')
                    
                    ff.write(f'For controller {name}:\n')

                    with open(data_path + metric + name + "_graphsize.pkl", 'rb') as file:
                        gs_dict = pickle.load(file)
                        
                    with open(data_path + metric + name + "_res.pkl", 'rb') as file:
                        res_dict = pickle.load(file)
                        
                        
                    count = 0
                    gs1 = 0.
                    gs2 = 0.
                    gs3 = 0.
                    for i in gs_dict.keys():
                        for gs in gs_dict[i]:
                            # before normalization non-zero/zero, after normalization non-zero
                            for (sz1, sz2, sz3) in gs:
                                gs1 += sz1
                                gs2 += sz2
                                gs3 += sz3
                                count += 1
                            
                        # print(f'For {i}, {count}')
                        
                    gs1 = gs1/count if count > 0 else 0.
                    gs2 = gs2/count if count > 0 else 0.
                    gs3 = gs3/count if count > 0 else 0.
                    
                    ff.write(f'The average graph is before normalization zero and nonzero edges are {gs2} and {gs1}, after normalization non-zero edges are {gs3}\n')
                    ff.write(f'remove edges {gs1-gs3}, remove edge ratio is {((gs1-gs3)/gs1) :.5f}...\n\n')
                        
 
       
                    count = 0
                    neg_c = 0.
                    topneg_c = 0.
                    total_e = 0.
                    frac_neg = 0.
                    frac_top = 0.
                    avg_c = 0.
                    med_c = 0.
                    low_c = 0.
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
                    for i in res_dict.keys():
                        for res in res_dict[i]:
                            for (ricci, batch, dim) in res:
                                neg, total, top, curv, curv_per_l = get_fraction(ricci, batch, dim)
                                neg_c += neg
                                topneg_c += top
                                total_e += total
                                frac_neg += (neg/total)
                                frac_top += (top/total)
                                
                                min_values = [min(sublist) for sublist in curv_per_l if sublist]
                                min_layer += np.array(min_values)
                                
                                c_num = len(curv)
                                curv = np.array(curv)
                                total_c += c_num
                                total_top_c += (len(curv[curv < -15])/c_num)
                                c_neg += len(curv[curv<0])
                                c_zero += len(curv[curv==0])
                                c_pos += len(curv[curv>0])
                                c_neg_f += (len(curv[curv<0])/c_num)
                                c_zero_f += (len(curv[curv==0])/c_num)
                                c_pos_f += (len(curv[curv>0])/c_num)
                                last_5per_c += curv[-(int)(0.04*c_num)]
                                avg_c += np.mean(curv)
                                med_c += np.median(curv)
                                low_c += np.min(curv)
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
                    
                    ff.write(f'For each layer the average total edge is {total_e}, the average negative edge is {neg_c}, the average fraction is {frac_neg}...\n')
                    ff.write(f'For each layer the average top-k (< -15) negative curvature edge is {topneg_c}, the average fraction is {frac_top}, total average fraction for curv < -15 is {total_top_c:.3f}...\n')
                    ff.write(f'The average total curvatrue number is {total_c}, postive is {c_pos} ({c_pos_f:.4f}), zero is {c_zero} ({c_zero_f:.4f}), negative is {c_neg} ({c_neg_f:.4f}). The last 4% curvature is {last_5per_c}\n\n')
                    ff.write(f'The average min value of each layer is {min_layer}\n\n')
                    
                    # ff.write(f'For last layer the average total edge is {total_e_last}, the average negavtive curvature edge {neg_c_last}, the average top-k (<-10) negavtive curvature edge is {topneg_c_last}...\n')
                    # ff.write(f'For last layer, the average negavtive curvature edge fraction {frac_neg_last}, the average top-k (<-10) negavtive curvature edge fraction {frac_top_last}...\n\n')

                    avg_c = avg_c/count if count > 0 else 0.
                    med_c = med_c/count if count > 0 else 0.
                    low_c = low_c/count if count > 0 else 0.
                    high_c = high_c/count if count > 0 else 0.
                    # last_10_percent_values = last_10_percent_values/count if count > 0 else 0.
                    var_c = var_c/count if count > 0 else 0.
                    
                    ff.write(f'For all the curvatures: \n')
                    ff.write(f'The average is {avg_c :.3f}, lowest is {low_c :.3f}, the highest is {high_c :.3f}, the median is {med_c :.3f}, variance is {var_c :.3f}\n\n')
                    # ff.write(f'The last 10 percent curvature average value is {last_10_percent_values :.3f}\n\n')
                    
                                
                                
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