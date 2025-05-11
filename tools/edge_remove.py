import torch
import numpy as np

class Edge_Remove():
    def __init__(self, net, dims, rem_n, file_path):
        self.net = net
        self.dims = dims
        self.layer_num = len(dims) - 1
        self.file_path = file_path
        self.num = rem_n
        self.predim = np.cumsum([0] + dims).tolist()
        
    def e_remove(self, edge_set, mark):
        self.net.eval()
        for i, (n1, n2) in enumerate(edge_set):
            if i >= (int)(self.num):
                break

            i_layer = np.searchsorted(self.predim, n1, side='right') - 1
            j_layer = np.searchsorted(self.predim, n2, side='right') - 1
            
            i_idx = n1 - self.predim[i_layer]
            j_idx = n2 - self.predim[j_layer]
       
            with torch.no_grad():
                if (j_layer == self.layer_num):
                    self.net.layer_list[j_layer-1].weight[j_idx,i_idx] = 0
                else: 
                    self.net.layer_list[j_layer-1].f4.f4.weight[j_idx,i_idx] = 0
                    
        torch.save(self.net.state_dict(), self.file_path + mark)
       