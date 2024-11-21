import torch
import torch.nn as nn
from collections import OrderedDict

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
    
    # calculate edge weights
    def edge_w(self, x):
        x = x.view(-1, self.in_dim)
        x_tmp = x
        
        for i in range(self.num_hidden_layers):  
            x = self.layer_list[i](x)       
            
            x_tmp = x_tmp.T * self.layer_list[i].f4.f4.weight.T
            x_tmp = torch.reshape(x_tmp, (x.shape[0], self.layer_list[i].f4.f4.weight.shape[0]*self.layer_list[i].f4.f4.weight.shape[1]))

            if i == 0:
                output = x_tmp
            else:
                output = torch.cat((output,x_tmp), axis=1)
            x_tmp = x
        
        x_tmp = x_tmp.T * self.layer_list[self.num_hidden_layers].weight.T
        x_tmp = torch.reshape(x_tmp, (1, self.layer_list[self.num_hidden_layers].weight.shape[0]*self.layer_list[self.num_hidden_layers].weight.shape[1]))

        x =  self.layer_list[self.num_hidden_layers](x)
        # x = self.sigmoid(x)
        
        output = torch.cat((output, x_tmp), axis=1)
        return output
    
    
    # calculate edge weights
    def edge_w_batch(self, x):
        x = x.view(-1, self.in_dim) # batch * input size
        x_tmp = x
        nodes = x
        nodes_before = torch.zeros_like(nodes)
        
        for i in range(self.num_hidden_layers):  
            x1 = x @ self.layer_list[i].f4.f4.weight.T
            nodes_before = torch.cat((nodes_before, x1), axis = 1)
            
            x = self.layer_list[i](x)      
            nodes = torch.cat((nodes, x), axis = 1)
            
            assert(nodes_before.shape == nodes.shape)
            
            w = self.layer_list[i].f4.f4.weight.T
            cur_shape = self.layer_list[i].f4.f4.weight.shape[0]*self.layer_list[i].f4.f4.weight.shape[1]
            
            x_tmp = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)

            if i == 0:
                output = x_tmp
            else:
                output = torch.cat((output,x_tmp), axis=1)
            x_tmp = x
            
            
        w = self.layer_list[self.num_hidden_layers].weight.T
        cur_shape = self.layer_list[self.num_hidden_layers].weight.shape[0]*self.layer_list[self.num_hidden_layers].weight.shape[1]
            
        x_tmp = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)

        x1 = x @ self.layer_list[self.num_hidden_layers].weight.T
        nodes_before = torch.cat((nodes_before, x1), axis = 1)
        
        
        x =  self.layer_list[self.num_hidden_layers](x)
        nodes = torch.cat((nodes, x), axis = 1)
        
        output = torch.cat((output, x_tmp), axis=1)
        
        assert(nodes_before.shape == nodes.shape)
        
        return output, nodes, nodes_before
    

    # get NN weights
    def get_weights(self, x1):
        x = torch.ones_like(x1)
        
        x = x.view(-1, self.in_dim) # batch * input size
        x_tmp = x
        for i in range(self.num_hidden_layers):  
            x = self.layer_list[i](x)      
            x = torch.ones_like(x) 
            
            w = self.layer_list[i].f4.f4.weight.T
            cur_shape = self.layer_list[i].f4.f4.weight.shape[0]*self.layer_list[i].f4.f4.weight.shape[1]
            
            x_tmp = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)

            if i == 0:
                output = x_tmp
            else:
                output = torch.cat((output,x_tmp), axis=1)
            x_tmp = x
            
        w = self.layer_list[self.num_hidden_layers].weight.T
        cur_shape = self.layer_list[self.num_hidden_layers].weight.shape[0]*self.layer_list[self.num_hidden_layers].weight.shape[1]
            
        x_tmp = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)

        x =  self.layer_list[self.num_hidden_layers](x)
        # x = self.sigmoid(x)
        x = torch.ones_like(x) 
        
        output = torch.cat((output, x_tmp), axis=1)
        
        return output
    
    
    def get_new_edge_v(self, nodes_v, adjacent_m):
        adj_new = adjacent_m * nodes_v[0].view(1, -1)
        return adj_new