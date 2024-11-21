import torch
import torch.nn as nn
import numpy as np

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx


''''
1. calculate curvature
2. get indices
3. using edge weights to calculate loss

'''
class CUR_LOSS(nn.Module):
    def __init__(self, net, dims, mask, mask_dict, target = None):
        super(CUR_LOSS, self).__init__()
        self.net = net
        self.dims = dims
        self.layer_num = len(dims) - 1
        self.nodes_num, self.edges_num = self.__get_net_status__()
        self.mask = mask
        self.mask_dict = mask_dict
        self.target = target
        
        
    # calculate nodes num and edges num for net  
    def __get_net_status__(self):
        neural_list = []
        nodes_num = 0
        edges_num = 0
        i = 0
        for p in self.net.parameters():
            if i == 0:
                nodes_num += p.shape[1]
            if i%2 == 0:
                nodes_num += p.shape[0]
                edges_num += (p.shape[0] * p.shape[1])
                neural_list.append(p.shape[0])
            i += 1
            
        return nodes_num, edges_num
         
           
    # build adjacent matric based on absoluate edge values
    def build_adj_mat(self, abs_edge_w):

        # build adjacent matrix
        adjacent_m = np.zeros((self.nodes_num, self.nodes_num), dtype=np.float32)

        layer_num = len(self.dims)
        cur_s_col = self.dims[0]
        cur_e_col = self.dims[0] + self.dims[1]

        cur_layer = 1
        start_col = 0
        end_col = self.dims[1]

        for i in range(self.nodes_num - self.dims[layer_num-1]):
            # print(f'i : {i}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
            adjacent_m[i, cur_s_col : cur_e_col] = abs_edge_w[start_col : end_col]
            
            if (cur_layer < layer_num-1 and i == cur_s_col - 1):
                cur_layer += 1
                start_col = end_col
                end_col = end_col + self.dims[cur_layer]
                cur_s_col = cur_e_col
                cur_e_col = cur_e_col + self.dims[cur_layer]
            else:
                start_col = end_col
                end_col = end_col + self.dims[cur_layer]
                    
        return adjacent_m
    
    
    # get top 1000 edges based on curvature
    def get_top_edge(self, adj_m):
        # build graph and calculate curvature
        G = nx.from_numpy_array(adj_m)

        orf = OllivierRicci(G, alpha=0.5, verbose="TRACE")
        orf.compute_ricci_curvature()

        G1 = orf.G.copy()
        
        # select top 1000 edges
        edge = sorted(G1.edges(data=True), key=lambda edge: edge[2].get('ricciCurvature', 0), reverse = True)[0:1000]
        edge = list(np.array(edge)[:, 0:2])
        my_set = {(n1,n2) for (n1,n2) in edge}
        return my_set
    
    
    # build mask
    def build_mask(self, mask, edge_set):
        
        for (n1, n2) in edge_set:
            cur_l = 0
            index = 0
            
            while (cur_l < self.layer_num and (n1 > (self.dims[cur_l]-1))):
                n1 -= self.dims[cur_l]
                n2 -= self.dims[cur_l]
                cur_l += 1
                index += (self.dims[cur_l] * self.dims[cur_l-1])
                
            assert(n2 >= 0)
            while (cur_l < self.layer_num and (n2 > (self.dims[cur_l]-1))):
                n2 -= self.dims[cur_l]
                cur_l += 1
                
            index += (n1*self.dims[cur_l] + n2)
            mask[:, index] = 1.
        
        return mask
        
    
    # start function to calculate the loss 
    def forward(self, images):
        edge_w = self.net.edge_w_batch(images) # [1, edge num]
        device = edge_w.device
        
        # with torch.no_grad():   
        #     edge_w_new = edge_w.cpu().detach().numpy()
        #     abs_edge_w = np.abs(edge_w_new) # absolate edge value for one image
        #     w_avg = np.mean(abs_edge_w, axis=0) # (edge num,)
        #     adj_m = self.build_adj_mat(w_avg)
            
        #     # top 1000 edges indices
        #     edge_set = self.get_top_edge(adj_m)
            
        # mask = torch.zeros(edge_w.shape, dtype=torch.float32)
        # mask = self.build_mask(mask, edge_set).to(device)
        
        mask = self.mask.to(device)
        
        if self.target == None:  
            loss = torch.sum(mask*torch.abs(edge_w))
        else:
            mask_target = torch.zeros(mask.shape, dtype=torch.int32)
            
            for t in self.target:
                mask_target |= self.mask_dict[t]
                
            mask_target = mask_target.to(device)
                
            tmp = mask & mask_target
            loss = torch.sum((mask - mask_target)* torch.abs(edge_w))
            
        return loss
            
            