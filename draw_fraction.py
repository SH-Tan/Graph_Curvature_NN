import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict
import statsmodels.api as sm
from scipy.integrate import simps

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

selected_classes = [0,1,2,3,4,5,6,7,8,9]
eps = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
Q = [1]

layer = [2]
res_path = 'img/w7/'
data_path = 'CNN/res/'
# fc_name = "frac_robust.pkl"
# fc_name_no = "frac_norobust.pkl"
cnn_name = '_frac_robust.pkl'
cnn_name_no = '_frac_norobust.pkl'

q_str = '_w7_'
mod_n = "cnn"

# str(e) + "_w5_" + str(q) + '_' + str(layer_num) + "frac_robust.pkl"

with open("./Frac_AUC_f_w7.txt", "a+") as ff:
    # Plot the data
    for layer_num in layer:
        model_n = mod_n # + str(layer_num)
        ff.write(f'Metric = {q_str}: Current model is {model_n}, layer num {layer_num}...\n')
        
        for q in Q:
            robust_area = defaultdict(list)
            norobust_area = defaultdict(list)
            
            for e in eps:
                name = str(e) + q_str + str(q) + cnn_name
                name1 = str(e) + q_str + str(q) + cnn_name_no
                
                # name = str(e) + q_str + str(q) + '_' + str(layer_num) + fc_name
                # name1 = str(e) + q_str + str(q) + '_' + str(layer_num) + fc_name_no
                
                with open(data_path + name, 'rb') as file:
                    robust_dict = pickle.load(file)
                
                num = 0.
                count = 0
                for l in selected_classes:
                    
                    for f in robust_dict[l]:
                        num += f
                        count += 1
                        
                    print(f'For {l}, eps = {e}, {count}')
                        
                        # ax.plot(x, y, color = 'red', linewidth = 1, label="robust img")
                        # ax.hist(d, bins=100, alpha=0.8, label=str(l), color = 'red')
                num = num/count if count > 0 else 0.
                robust_area[e].append(num)
                    
    
                with open(data_path + name1, 'rb') as file:
                    norobust_dict = pickle.load(file)
                
                
                num = 0.
                count = 0
                for l in selected_classes:
                    for f in norobust_dict[l]:
                        num += f
                        count += 1
                        
                        # ax.plot(x, y, color = 'red', linewidth = 1, label="robust img")
                        # ax.hist(d, bins=100, alpha=0.8, label=str(l), color = 'red')
                num = num/count if count > 0 else 0.
                norobust_area[e].append(num)
                    
                ff.write(f'For eps = {e}, the average fraction of robust images is {robust_area[e]}, of the non-robust images is {norobust_area[e]}...\n\n')

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