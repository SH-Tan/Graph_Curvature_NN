import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

selected_classes = [0,1,2,3,4,5,6,7,8,9]
eps = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
Q = [1]

layer = [2,4]
# res_path = 'img/w1/'
data_path = 'pgd/res_decay/'

fc_name_auc = "_robust.pkl"
fc_name_no_auc = "_norobust.pkl"
fc_name_frac = "frac_robust.pkl"
fc_name_no_frac = "frac_norobust.pkl"

# cnn_name_auc = '_robust_norm10.pkl'
# cnn_name_no_auc = '_norobust_norm10.pkl'
# cnn_name_frac = '_frac_robust_norm10.pkl'
# cnn_name_no_frac = '_frac_norobust_norm10.pkl'

q_str = '_w7_'
mod_n = "fc_decay"
mark = "_ori_c_"

ranges = [-500, -200, -100, -50, -20, -10, -5, -2, -1, -0.5, -0.1, -0.01, 0., 0.1, 0.5, 1.]


with open("./Frac_AUC_w7.txt", "a+") as ff:
    ff.write(f'The range to calculate the fraction is {ranges}..\n\n')
    # Plot the data
    for layer_num in layer:
        model_n = mod_n # + str(layer_num)
        ff.write(f'Metric = {q_str}: Current model is {model_n}, layer num {layer_num}...\n\n')
        
        for q in Q:
            robust_area = defaultdict(list)
            norobust_area = defaultdict(list)
            
            for e in eps:
                # c_name = q_str + str(q) + mark + str(e) + cnn_name_auc
                # c_name_no = q_str + str(q) + mark + str(e) + cnn_name_no_auc
                
                c_name = q_str + str(q) + mark + str(e) + '_' + str(layer_num) + fc_name_auc
                c_name_no = q_str + str(q) + mark + str(e) + '_' + str(layer_num) + fc_name_no_auc
                
                with open(data_path + c_name, 'rb') as file:
                    robust_dict = pickle.load(file)
                
                f = np.zeros(len(ranges), dtype=np.float64)
                count = 0
                
                one = 0.
                zero = 0.
                
                for l in selected_classes:
                    for c in robust_dict[l]:
                        total_e = c.shape[0]
                        
                        one += len(c[c==1])/total_e
                        zero += len(c[c==0])/total_e
                        
                        for r in range(len(ranges)):
                            frac = (c[c<ranges[r]].shape[0])/total_e
                            f[r] += frac
                        
                        count += 1
                        
                    # print(f'For {l}, eps = {e}, {count}')
                        
                f = f/count if count > 0 else f
                zero = zero/count if count > 0 else zero
                one = one/count if count > 0 else one
                # robust_area[e].append(f)
                
                ff.write(f'For eps = {e}, the average fraction of robust images is {f}...\n')
                ff.write(f'For eps = {e}, zero curvature edge ratio is {zero}, one curvature edge ratio is {one}...\n\n')
                    
                    
                # with open(data_path + c_name_no, 'rb') as file:
                #     norobust_dict = pickle.load(file)
                
                
                # f = np.zeros(len(ranges), dtype=np.float64)
                # count = 0
                
                # one = 0.
                # zero = 0.
                
                # for l in selected_classes:
                #     for c in norobust_dict[l]:
                #         total_e = c.shape[0]
                        
                #         one += len(c[c==1])/total_e
                #         zero += len(c[c==0])/total_e
                        
                #         for r in range(len(ranges)):
                #             frac = (c[c<ranges[r]].shape[0])/total_e
                #             f[r] += frac
                        
                #         count += 1
                        
                # f = f/count if count > 0 else f
                # zero = zero/count if count > 0 else zero
                # one = one/count if count > 0 else one
                    
                # ff.write(f'For eps = {e}, the average fraction of non-robust images is {f}...\n')
                # ff.write(f'For eps = {e}, zero curvature edge ratio is {zero}, one curvature edge ratio is {one}...\n\n')
                    
                    

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
            # plt.savefig(res_path + model_n + str(e) + q_str + str(layer_num) + "_frac.png")
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