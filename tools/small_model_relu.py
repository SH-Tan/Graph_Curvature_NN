import torch
import torch.nn as nn
from collections import OrderedDict
from .q_exponential import q_exponential
import numpy as np


class FC1(nn.Module):
    def __init__(self, input_s, output_s):
        super(FC1, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(input_s, output_s)),
            ('relu4', nn.ReLU())
        ]))
        
    def forward(self, img):
        output = self.f4(img)
        return output
    

class FC_MD(nn.Module):

    def __init__(self, dims, num_hidden_layers):
        super(FC_MD, self).__init__()
        
        assert (len(dims) == num_hidden_layers + 2)

        self.in_dim = dims[0]
        self.out_dim = dims[num_hidden_layers + 1]
        
        self.layer_list = nn.ModuleList()

        self.num_hidden_layers = num_hidden_layers
        
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

        for i in range(0, self.num_hidden_layers):
            self.layer_list.append(FC1(dims[i], dims[i+1]))
            
        self.layer_list.append(nn.Linear(dims[num_hidden_layers], self.out_dim))
        

    def forward(self, x):
        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = self.layer_list[i](x)
            
        y = self.layer_list[self.num_hidden_layers](x)
        return y
    
    
    
    
    # return the activation value (node value) of each neural
    def activation(self, x):
        x = x.view(-1, self.in_dim)
        
        for i in range(self.num_hidden_layers):          
            x = self.layer_list[i](x)
            if i == 0:
                output = x
            else:
                output = torch.cat((output,x), axis=1)
            
        x =  self.layer_list[self.num_hidden_layers](x)
        # x = self.sigmoid(x)
        
        output = torch.cat((output,x), axis=1)
        return output
    
    
    
    
    # calculate edge weights, NN weights, node value (before/ after ReLU)
    def NN_info_batch(self, x):
        x = x.view(-1, self.in_dim) # batch * input size
        x_tmp = x
        node_tmp = x
        
        nodes = x

        ones_tmp = torch.ones_like(x)
        
        for i in range(self.num_hidden_layers):  
            # ndoes after activation
            x = self.layer_list[i](x)      
            nodes = torch.cat((nodes, x), axis = 1)
            
            # edges
            w = self.layer_list[i].f4.f4.weight.T
            cur_shape = self.layer_list[i].f4.f4.weight.shape[0]*self.layer_list[i].f4.f4.weight.shape[1]
            w_tmp = torch.ones_like(w)
            
            x_tmp = torch.cat([torch.reshape(w * xx[np.newaxis,:].T, (1, cur_shape)) for xx in x_tmp], axis=0)
            ones_tmp = torch.cat([torch.reshape(w * xx[np.newaxis,:].T, (1, cur_shape)) for xx in ones_tmp], axis=0)
            node_tmp = torch.cat([torch.reshape(w_tmp * xx[np.newaxis,:].T, (1, cur_shape)) for xx in node_tmp], axis=0)

            output = x_tmp if i == 0 else torch.cat((output,x_tmp), axis=1)
            weights = ones_tmp if i == 0 else torch.cat((weights,ones_tmp), axis=1)
            all_node = node_tmp if i == 0 else torch.cat((all_node,node_tmp), axis=1)
   
            x_tmp = x
            node_tmp = x
            ones_tmp = torch.ones_like(x) 
            
        # last layer
        w = self.layer_list[self.num_hidden_layers].weight.T
        cur_shape = self.layer_list[self.num_hidden_layers].weight.shape[0]*self.layer_list[self.num_hidden_layers].weight.shape[1]
        w_tmp = torch.ones_like(w)
        
        x_tmp = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)
        ones_tmp = torch.cat([torch.reshape(w * xx[np.newaxis,:].T, (1, cur_shape)) for xx in ones_tmp], axis=0)
        node_tmp = torch.cat([torch.reshape(w_tmp * xx[np.newaxis,:].T, (1, cur_shape)) for xx in node_tmp], axis=0)

        output = torch.cat((output, x_tmp), axis=1)
        weights = torch.cat((weights,ones_tmp), axis=1)
        all_node = node_tmp if i == 0 else torch.cat((all_node,node_tmp), axis=1)
        
        # output layer nodes
        x1 = x @ self.layer_list[self.num_hidden_layers].weight.T
         
        x =  self.layer_list[self.num_hidden_layers](x)
        nodes = torch.cat((nodes, x), axis = 1)
        
        return output, nodes, weights, all_node
    

    
    
    '''
    weights: batch * edge num
    nodes: batch * node num
    '''
    def build_adj(self, nodes, weights, dims):
        batch = nodes.shape[0]
        nodes_num = nodes.shape[1]
        
        # build adjacent matrix
        adjacent_m = torch.zeros((batch, nodes_num, nodes_num), dtype=torch.float32)

        d_num = len(dims)
        cur_s_col = dims[0]
        cur_e_col = dims[0] + dims[1]

        cur_layer = 1
        start_col = 0
        end_col = dims[1]

        for i in range(nodes_num - dims[d_num-1]):
            # print(f'i : {i}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
            adjacent_m[:, i, cur_s_col : cur_e_col] = weights[:, start_col : end_col]
            
            if (cur_layer < d_num-1 and i == cur_s_col - 1):
                cur_layer += 1
                start_col = end_col
                end_col = end_col + dims[cur_layer]
                cur_s_col = cur_e_col
                cur_e_col = cur_e_col + dims[cur_layer]
            else:
                start_col = end_col
                end_col = end_col + dims[cur_layer]
                
        return adjacent_m
    
    
    
    
    # weights regularization 
    def recal_graph_weight_w2(self, nodes, adj, dims):
        batch = nodes.shape[0]
        nodes_num = nodes.shape[1]
        prefix_dims = np.cumsum(dims)
        cur_layer = 0
        
        adj_inv = torch.zeros_like(adj)
        # adj_noninv = torch.zeros_like(adj)
         
        for n in range(dims[0], nodes_num):
            if (n >= prefix_dims[cur_layer]):
                cur_layer += 1
            
            start = 0 if cur_layer == 1 else prefix_dims[cur_layer-2]
            neighbors = torch.arange(start, prefix_dims[cur_layer-1])
            
            w = nodes[:, neighbors]*adj[:, neighbors, n]
            
            positive_w = torch.where(w > 0, w, torch.zeros_like(w))
            pos_sum = positive_w.sum(dim=1, keepdim=True)
            
            sum = torch.sum(w, axis = 1, keepdim=True)
            
            positive_s_i = torch.where(sum > 0)[0]
            # negative_s_i = torch.where(sum <= 0)[0]
            
            sub_pos_a = adj[positive_s_i][:, neighbors][:, :, n]
                        
            mask = sub_pos_a > 0
            values = (sub_pos_a * (sum[positive_s_i] / pos_sum[positive_s_i]) * nodes[positive_s_i][:, neighbors])
            
            sub_pos_a_inv = torch.where(mask, 1./values, torch.tensor(0.))
            # sub_pos_a = torch.where(mask, values, torch.tensor(0.))

            # adj_noninv[torch.tensor(positive_s_i)[:,None], torch.tensor(neighbors)[None], torch.tensor(n)] = sub_pos_a
            adj_inv[torch.tensor(positive_s_i)[:,None], torch.tensor(neighbors)[None], torch.tensor(n)] = sub_pos_a_inv
    
        return adj_inv.to_sparse()
    
    
    
    # weights regularization 
    def normalization_weight_w1(self, nodes, weights, dims):
        # batch = nodes.shape[0]
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims])

        cur_layer = 1
        start_col = 0
        end_col = dims[cur_layer-1] * dims[cur_layer]
        step = dims[cur_layer]
        neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
        
        weights_inv1 = torch.zeros_like(weights)
        weights_inv2 = torch.zeros_like(weights)
        
        # go through each node except input layer
        for n in range(dims[0], nodes_num):
            if (n >= prefix_dims[cur_layer+1]):
                cur_layer += 1
                
                start_col = end_col
                end_col += (dims[cur_layer-1] * dims[cur_layer])
                step = dims[cur_layer]
            
                neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
            
            # print(f'start: {start_col + (n - prefix_dims[cur_layer])}, end: {end_col}, step: {step}')
            in_edges = torch.arange(start_col + (n - prefix_dims[cur_layer]), end_col, step)
            
            w = nodes[:, neighbors] * weights[:, in_edges] # batch * neighbors.size()
            
            positive_w = torch.where(w > 0, w, torch.zeros_like(w))
            pos_sum = positive_w.sum(dim=1, keepdim=True) # batch * 1
            
            sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
            
            positive_s_i = torch.where(sum > 0)[0] # indices
            
            sub_pos_a1 = weights[positive_s_i][:, in_edges]   
            sub_pos_a2 = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
                    
            mask = sub_pos_a1 > 0
            values1 = sub_pos_a1 * (sum[positive_s_i] / pos_sum[positive_s_i])
            values2 = torch.abs(sub_pos_a2 * (sum[positive_s_i] / pos_sum[positive_s_i]))
            
            sub_pos_a_inv1 = torch.where(mask, 1./values1, torch.tensor(0.))
            sub_pos_a_inv2 = torch.where(mask, 1./values2, torch.tensor(0.))
            weights_inv1[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv1
            weights_inv2[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv2

        return weights_inv1, weights_inv2
    
    
    
    def normalization_weight_w2_old(self, nodes, weights, dims):
        # batch = nodes.shape[0]
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims])

        cur_layer = 1
        start_col = 0
        end_col = dims[cur_layer-1] * dims[cur_layer]
        step = dims[cur_layer]
        neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
        
        # weights_inv1 = torch.zeros_like(weights)
        weights_inv2 = torch.zeros_like(weights)
        
        # go through each node except input layer
        for n in range(dims[0], nodes_num):
            if (n >= prefix_dims[cur_layer+1]):
                cur_layer += 1
                
                start_col = end_col
                end_col += (dims[cur_layer-1] * dims[cur_layer])
                step = dims[cur_layer]
            
                neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
            
            # print(f'start: {start_col + (n - prefix_dims[cur_layer])}, end: {end_col}, step: {step}')
            in_edges = torch.arange(start_col + (n - prefix_dims[cur_layer]), end_col, step)
            
            w = nodes[:, neighbors] * weights[:, in_edges] # batch * neighbors.size()
            
            positive_w = torch.where(w > 0, w, torch.zeros_like(w))
            pos_sum = positive_w.sum(dim=1, keepdim=True) # batch * 1
            
            sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
            
            positive_s_i = torch.where(sum > 0)[0] # indices
            
            # sub_pos_a1 = weights[positive_s_i][:, in_edges]   
            sub_pos_a2 = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
                    
            mask = sub_pos_a2 > 0
            # values1 = (sub_pos_a1 * (sum[positive_s_i] / pos_sum[positive_s_i]))
            values2 = (sub_pos_a2 * (sum[positive_s_i] / pos_sum[positive_s_i]))
            
            # sub_pos_a_inv1 = torch.where(mask, values2, torch.tensor(0.))
            sub_pos_a_inv2 = torch.where(mask, 1./values2, torch.tensor(0.))
            # weights_inv1[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv1
            weights_inv2[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv2

        return weights_inv2
    
    
    
    def normalization_weight_w2(self, nodes, weights, dims):
        # batch = nodes.shape[0]
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims])

        cur_layer = 1
        start_col = 0
        end_col = dims[cur_layer-1] * dims[cur_layer]
        step = dims[cur_layer]
        neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
        
        # weights_inv_neg = torch.zeros_like(weights)
        weights_inv = torch.zeros_like(weights)
        
        # go through each node except input layer
        for n in range(dims[0], nodes_num):
            if (n >= prefix_dims[cur_layer+1]):
                cur_layer += 1
                
                start_col = end_col
                end_col += (dims[cur_layer-1] * dims[cur_layer])
                step = dims[cur_layer]
            
                neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
            
            # print(f'start: {start_col + (n - prefix_dims[cur_layer])}, end: {end_col}, step: {step}')
            in_edges = torch.arange(start_col + (n - prefix_dims[cur_layer]), end_col, step)
            
            w = nodes[:, neighbors] * weights[:, in_edges] # batch * neighbors.size()
            
            positive_w = torch.where(w > 0, w, torch.zeros_like(w))
            pos_sum = positive_w.sum(dim=1, keepdim=True) # batch * 1
            
            negative_w = torch.where(w <= 0, w, torch.zeros_like(w))
            neg_sum = negative_w.sum(dim=1, keepdim=True) # batch * 1
            
            sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
            
            positive_s_i = torch.where(sum > 0)[0] # indices
            negative_s_i = torch.where(sum <= 0)[0] # indices
            
            # sub_pos_a1 = weights[positive_s_i][:, in_edges]   
            sub_pos_a = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
            sub_neg_a = weights[negative_s_i][:, in_edges] * nodes[negative_s_i][:, neighbors]
                    
            mask_pos = sub_pos_a > 0
            mask_neg = sub_neg_a <= 0

            values_pos = (sub_pos_a * (sum[positive_s_i] / pos_sum[positive_s_i]))
            values_neg = -(sub_neg_a * (sum[negative_s_i] / neg_sum[negative_s_i]))
            
            sub_pos_a_inv = torch.where(mask_pos, 1./values_pos, torch.tensor(0.))
            sub_neg_a_inv = torch.where(mask_neg, 1./values_neg, torch.tensor(0.))
            
            weights_inv[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv
            weights_inv[torch.tensor(negative_s_i)[:,None], torch.tensor(in_edges)] = sub_neg_a_inv

        return weights_inv

    
    
    # weights regularization 
    def normalization_weight_w3(self, nodes, weights, dims):
        # batch = nodes.shape[0]
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims])

        cur_layer = 1
        start_col = 0
        end_col = dims[cur_layer-1] * dims[cur_layer]
        step = dims[cur_layer]
        neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
        
        weights_inv1 = torch.zeros_like(weights)
        weights_inv2 = torch.zeros_like(weights)
        
        # go through each node except input layer
        for n in range(dims[0], nodes_num):
            if (n >= prefix_dims[cur_layer+1]):
                cur_layer += 1
                
                start_col = end_col
                end_col += (dims[cur_layer-1] * dims[cur_layer])
                step = dims[cur_layer]
            
                neighbors = torch.arange(prefix_dims[cur_layer-1], prefix_dims[cur_layer])
            
            # print(f'start: {start_col + (n - prefix_dims[cur_layer])}, end: {end_col}, step: {step}')
            in_edges = torch.arange(start_col + (n - prefix_dims[cur_layer]), end_col, step)
            
            w = nodes[:, neighbors] * weights[:, in_edges] # batch * neighbors.size()
            
            positive_w = torch.where(w > 0, w, torch.zeros_like(w))
            pos_sum = positive_w.sum(dim=1, keepdim=True) # batch * 1
            
            negative_w = torch.where(w <= 0, w, torch.zeros_like(w))
            neg_sum = negative_w.sum(dim=1, keepdim=True) # batch * 1
            
            Sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
            
            positive_s_i = torch.where(Sum > 0)[0] # indices
            negative_s_i = torch.where(Sum <= 0)[0] # indices
            
            # w2 - probability
            sub_pos_a2 = torch.ones_like(weights[positive_s_i][:, in_edges]) * nodes[positive_s_i][:, neighbors]
            sub_neg_a2 = torch.ones_like(weights[negative_s_i][:, in_edges]) * nodes[negative_s_i][:, neighbors]
            
            # w1
            sub_pos_a1 = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
            sub_neg_a1 = weights[negative_s_i][:, in_edges] * nodes[negative_s_i][:, neighbors]
            sub_pos = weights[positive_s_i][:, in_edges]
            sub_neg = weights[negative_s_i][:, in_edges]
            
            # w1
            mask_pos = sub_pos_a1 >= 0
            mask_neg = sub_neg_a1 <= 0

            values_pos = torch.abs(sub_pos * (Sum[positive_s_i] / pos_sum[positive_s_i]))
            values_neg = torch.abs(sub_neg * (Sum[negative_s_i] / neg_sum[negative_s_i]))
            
            sub_pos_a_inv = torch.where(mask_pos, 1./values_pos, torch.tensor(0.))
            sub_neg_a_inv = torch.where(mask_neg, 1./values_neg, torch.tensor(0.))
            
            weights_inv1[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv
            weights_inv1[torch.tensor(negative_s_i)[:,None], torch.tensor(in_edges)] = sub_neg_a_inv
            
            # w2
            values_pos2 = torch.abs(sub_pos_a2 * (Sum[positive_s_i] / pos_sum[positive_s_i]))
            values_neg2 = torch.abs(sub_neg_a2 * (Sum[negative_s_i] / neg_sum[negative_s_i]))
            
            sub_pos_a_inv2 = torch.where(mask_pos, 1./values_pos2, torch.tensor(0.))
            sub_neg_a_inv2 = torch.where(mask_neg, 1./values_neg2, torch.tensor(0.))

            weights_inv2[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv2
            weights_inv2[torch.tensor(negative_s_i)[:,None], torch.tensor(in_edges)] = sub_neg_a_inv2

        return weights_inv1, weights_inv2
    

    
    def normalization_weight_w6(self, nodes, weights, dims, q):
        # batch = nodes.shape[0]
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims])

        cur_layer = 1
        start_col = 0
        end_col = dims[cur_layer-1] * dims[cur_layer]
        step = dims[cur_layer]
        
        q_exp = q_exponential(q)
        weights_inv = torch.zeros_like(weights)
        
        # go through each node except input layer
        for n in range(dims[0], nodes_num):
            if (n >= prefix_dims[cur_layer+1]):
                cur_layer += 1
                
                start_col = end_col
                end_col += (dims[cur_layer-1] * dims[cur_layer])
                step = dims[cur_layer]
            
            # print(f'start: {start_col + (n - prefix_dims[cur_layer])}, end: {end_col}, step: {step}')
            in_edges = torch.arange(start_col + (n - prefix_dims[cur_layer]), end_col, step)
            
            w = weights[:, in_edges]
            mask = (w != 0)
            if nodes[:, n].item() <= 0:
                weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] = 1.0/abs(q_exp.q_exponential_series(-w))
            else:
                weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] = 1.0/abs(q_exp.q_exponential_series(w))
            
            weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] *= mask

        return weights_inv
    


    


