import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class XOR(nn.Module):

    def __init__(self, in_dim = 2, out_dim = 1, num_hidden_layers = 1, layer_size = 2):
        super(XOR, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layer_size = layer_size
        
        self.W1 = torch.tensor([[1,1],[0,1],[1,1],[1,5]], dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor([[0,-1,0,0]], dtype=torch.float32, requires_grad=True)
        self.w = torch.tensor([[1,-2,0,1]], dtype=torch.float32, requires_grad=True)

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1, self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))

    def forward(self, x):
        y1 = F.relu(x @ self.W1.T + self.b)
        y = y1 @ self.w.T
        return y
    
    
    # return the activation value of each neural
    def activation(self, x):
        y1 = F.relu(x @ self.W1.T + self.b)
        y = self.w @ y1
        
        output = y1.T
        output = torch.cat((output,y), axis=1)

        return output
    
    
    # calculate edge weights
    def edge_w_batch(self, x):
        x = x.view(-1, self.in_dim) # batch * input size
        x_tmp = x
        
        cur_shape = self.W1.shape[0]*self.W1.shape[1]
        x_tmp = torch.cat([torch.reshape(self.W1.T * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)

        output = x_tmp
        
        x = F.relu(x @ self.W1.T + self.b)
        x_tmp = x

        cur_shape = self.w.shape[0]*self.w.shape[1]
        x_tmp = torch.cat([torch.reshape(self.w.T * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)

        output = torch.cat((output, x_tmp), axis=1)
        
        return output