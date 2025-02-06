import torch
import torch.nn as nn
from collections import OrderedDict
from .q_exponential import q_exponential
import numpy as np


class FC1(nn.Module):
    def __init__(self, input_s, output_s):
        super(FC1, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(input_s, output_s))
            # ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output
    

class FC_Linear(nn.Module):

    def __init__(self, dims, num_hidden_layers):
        super(FC_Linear, self).__init__()
        
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
    
    
    # return the activation value of each neural
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
        
        nodes = x
        
        ones_tmp = torch.ones_like(x)
        
        for i in range(self.num_hidden_layers):  
            # ndoes after activation
            x = self.layer_list[i](x)      
            nodes = torch.cat((nodes, x), axis = 1)
            
            # edges
            w = self.layer_list[i].f4.f4.weight.T
            cur_shape = self.layer_list[i].f4.f4.weight.shape[0]*self.layer_list[i].f4.f4.weight.shape[1]
            
            x_tmp = torch.cat([torch.reshape(w * xx[np.newaxis,:].T, (1, cur_shape)) for xx in x_tmp], axis=0)
            ones_tmp = torch.cat([torch.reshape(w * xx[np.newaxis,:].T, (1, cur_shape)) for xx in ones_tmp], axis=0)

            output = x_tmp if i == 0 else torch.cat((output,x_tmp), axis=1)
            weights = ones_tmp if i == 0 else torch.cat((weights,ones_tmp), axis=1)
   
            x_tmp = x
            ones_tmp = torch.ones_like(x) 
            
        # last layer
        w = self.layer_list[self.num_hidden_layers].weight.T
        cur_shape = self.layer_list[self.num_hidden_layers].weight.shape[0]*self.layer_list[self.num_hidden_layers].weight.shape[1]
            
        x_tmp = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)
        ones_tmp = torch.cat([torch.reshape(w * xx[np.newaxis,:].T, (1, cur_shape)) for xx in ones_tmp], axis=0)

        output = torch.cat((output, x_tmp), axis=1)
        weights = torch.cat((weights,ones_tmp), axis=1)
         
        x =  self.layer_list[self.num_hidden_layers](x)
        nodes = torch.cat((nodes, x), axis = 1)
        
        return output, nodes, weights
    
    
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
            
            sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
            
            positive_s_i = torch.where(sum > 0)[0] # indices
            
            sub_pos_a = weights[positive_s_i][:, in_edges]
                        
            mask = sub_pos_a > 0
            values = (sub_pos_a * (sum[positive_s_i] / pos_sum[positive_s_i]))
            
            sub_pos_a_inv = torch.where(mask, 1./values, torch.tensor(0.))

            weights_inv[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv

        return weights_inv
    
    
    
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
            
            sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
            
            positive_s_i = torch.where(sum > 0)[0] # indices
            
            sub_pos_a = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
                        
            mask = sub_pos_a > 0
            values = (sub_pos_a * (sum[positive_s_i] / pos_sum[positive_s_i]))
            
            sub_pos_a_inv = torch.where(mask, 1./values, torch.tensor(0.))

            weights_inv[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv

        return weights_inv
    
    
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
    


    


