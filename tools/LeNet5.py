import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LeNet(nn.Module):

    # network structure
    def __init__(self, input_c = 1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_c, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.activation = nn.ReLU()
        

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(self.activation(self.conv1(x)), (2, 2))
        x = F.max_pool2d(self.activation(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)
    
    
    # calculate one cnn layer edge values
    def calculate_edge_v(self, input, w):
        k = w.unsqueeze(0).transpose(2, 0)
        edges = input.unsqueeze(1) * k[None,:,:,:]
        edges = edges.transpose(2,3)
        
        dim = edges.shape[1]*edges.shape[2]*edges.shape[3]
        edges = edges.reshape(-1, dim) # batch * edge_num
        
        return edges
        
    
    # CNN using unfold/fold, calculate edges values
    def CNN_edges(self, ori, kernel):
        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]))
        
        w = kernel.view(kernel.size(0),-1).T
        edge_v = self.calculate_edge_v(ori_unf, w)

        return edge_v
    
    
    # Pooling
    def pooling_edges(self, input, k_size = 2, stride = 2):
        #print(input.shape)
        batch = input.shape[0]
        channel = input.shape[1]

        i_unf = F.unfold(input, (k_size, k_size), stride=stride) 

        i_unf = i_unf.view(batch, channel, k_size*k_size, -1)
        
        edges = i_unf.transpose(2,3)

        dim = edges.shape[1]*edges.shape[2]*edges.shape[3]
        edges = edges.reshape(-1, dim) # batch * edge_num

        return edges
    
    # fullyconnected
    def fc_edges(self, x_tmp, layer):
        w = layer.weight.T
        cur_shape = layer.weight.shape[0]*layer.weight.shape[1]
        
        edge_v = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)
        return edge_v
        
    
    # calculate edge weights
    def edge_w_batch(self, x):
        edge_value = None
        
        # first CNN layer
        k1 = self.conv1.weight
        edge_v = self.CNN_edges(x, k1)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.conv1(x))
        
        # first max pooling
        edge_v = self.pooling_edges(x)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = F.max_pool2d(x, (2, 2))
        
        # second CNN layer
        k2 = self.conv2.weight
        edge_v = self.CNN_edges(x, k2)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.conv2(x))
        
        # second max pooling
        edge_v = self.pooling_edges(x)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = F.max_pool2d(x, (2, 2))
        
        # fully connected
        x = x.view(-1, self.num_flat_features(x)) # batch * input size
        x_tmp = x
        
        # fc1
        edge_v = self.fc_edges(x_tmp, self.fc1)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.fc1(x))
        x_tmp = x
        
        # fc2    
        edge_v = self.fc_edges(x_tmp, self.fc2)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.fc2(x))
        x_tmp = x
        
        # fc3
        edge_v = self.fc_edges(x_tmp, self.fc3)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.fc3(x)
        
        return edge_value