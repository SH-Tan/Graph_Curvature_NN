import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

types = ['DDPG', 'TD3']
cs = ['1','2','3']
szs = ['64','128']

# res_path = 'img/w1/'

metric = "w7"
data_path = "res/" + metric + "/"
# data_path = "res/w6_c/"

ranges = [-500, -200, -100, -50, -20, -10, -5, -2, -1, -0.5, -0.1, -0.01, 0., 0.1, 0.5, 1.]

samples = 1

with open("./Frac_AUC_w7.txt", "w+") as ff:
    ff.write(f'The range to calculate the fraction is {ranges}..\n\n')
    # Plot the data
    for t in types:
        for s in szs:
            for c in cs:
                name = '_'.join([t,s,c])
                
                # if t == 'DDPG' and s =='64' and c == '1':
                #     continue
                
                # if t == 'DDPG' and s == '128' and c == '3':
                #     continue
                
                model_n = metric + name + "_curvature.pkl"
                ff.write(f'Metric = {metric}: Current controller is {name}...\n\n')

                with open(data_path + model_n, 'rb') as file:
                    curv_dict = pickle.load(file)
                
                
                for i in range(samples):
                    f_safe = np.zeros(len(ranges), dtype=np.float64)
                    count = 0
                    
                    one = 0.
                    zero = 0.
                    around_z = 0.
                    
                    for c in curv_dict[i][0]:
                        total_e = c.shape[0]

                        one += len(c[c==1])/total_e
                        zero += len(c[c==0])/total_e
                        around_z += (len(c[c <= 0.1]) - len(c[c < -0.1]))/total_e
                            
                        for r in range(len(ranges)):
                            frac = (c[c<ranges[r]].shape[0])/total_e
                            f_safe[r] += frac
                            
                        count += 1
                            
                    f_safe = f_safe/count if count > 0 else f_safe
                    zero = zero/count if count > 0 else zero
                    one = one/count if count > 0 else one
                    around_z = around_z/count if count > 0 else around_z
                    
                    ff.write(f'The average fraction of safe trajectory is {f_safe}...\n')
                    ff.write(f'zero curvature edge ratio is {zero}, one curvature edge ratio is {one}, around zero is {around_z}...\n\n')
                
                
                # f_unsafe = np.zeros(len(ranges), dtype=np.float64)
                # count = 0
                # one = 0.
                # zero = 0.
                
                # for c in curv_dict['1'][0]:
                #     total_e = c.shape[0]
                    
                #     one += len(c[c==1])/total_e
                #     zero += len(c[c==0])/total_e
                        
                #     for r in range(len(ranges)):
                #         frac = (c[c<ranges[r]].shape[0])/total_e
                #         f_unsafe[r] += frac
                    
                #     count += 1
                        
                # f_unsafe = f_unsafe/count if count > 0 else f_unsafe
                # zero = zero/count if count > 0 else zero
                # one = one/count if count > 0 else one
                
                # ff.write(f'The average fraction of unsafe trajectory is {f_unsafe}...\n')
                # ff.write(f'zero curvature edge ratio is {zero}, one curvature edge ratio is {one}...\n\n')
                
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