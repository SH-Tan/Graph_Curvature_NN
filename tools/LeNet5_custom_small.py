import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class LeNet_custom_v2(nn.Module):

    # network structure
    def __init__(self, model_info, edge_set, device, input_c = 1):
        super(LeNet_custom_v2, self).__init__()
        self.conv1 = nn.Conv2d(input_c, 6, 6, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 6, stride=2)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.activation = nn.ReLU()
        
        self.model_info = model_info
        self.edge_set = edge_set if edge_set != None else set()
        self.cur_total_nodes = 0
        self.device = device
        self.remove_mask = dict()
        self.__build_remove_mask__(self.edge_set)
        
    
    def get_layer_info(self, l):
        cur_name = self.model_info[l]["name"]
        cur_dim = self.model_info[l]["dim"]
        cur_size = cur_dim['out_size']
        
        cur_channel = 1 if (cur_name == "fc") else cur_dim["channel"]
        
        if cur_name == "input":
            cur_nodes = cur_channel * cur_size**2
        else:
            cur_nodes = cur_size if (cur_name == "fc") else cur_dim["channel"]*(cur_size**2)
            
        return cur_nodes, cur_size, cur_channel, cur_dim, cur_name
    
    
    def __build_remove_mask__(self, new_edge_set = set()):
        cur_layer = 1
        self.cur_total_nodes = 0
        
        while(cur_layer < len(self.model_info)):
            # get l1, l2 info
            l1_nodes, l1_size, l1_channel, l1_dim, l1_name = self.get_layer_info(cur_layer)
            l2_nodes, l2_size, l2_channel, l2_dim, l2_name = self.get_layer_info(cur_layer+1)
            
            remove_e = [e for e in new_edge_set if (e[1] < (l2_nodes + l1_nodes + self.cur_total_nodes) and (e[1] >= l1_nodes + self.cur_total_nodes)) \
                and (e[0] >= self.cur_total_nodes and e[0] < (l1_nodes + self.cur_total_nodes))]

            if l2_name == "cnn":
                print("CNN remove = ", len(remove_e))
                k = l2_dim["kernel"]
                s = l2_dim["stride"]
                
                tensor_2d = torch.arange(l1_nodes).reshape(1,l1_channel,l1_size,l1_size).float()
        
                tensor_2d = torch.arange(l1_nodes).reshape(1,l1_channel,l1_size,l1_size).float()
                input_indices = F.unfold(tensor_2d, (k,k), stride = s).transpose(1,2).int()
                
                map_size = l2_size**2
                
                if cur_layer not in self.remove_mask.keys():
                    self.remove_mask[cur_layer] = torch.ones((l2_channel, input_indices.shape[1], input_indices.shape[2]))
                
                if (len(remove_e) > 0):
                    for e in remove_e:
                        n1 = e[0] - self.cur_total_nodes
                        n2 = e[1] - self.cur_total_nodes - l1_nodes
                        
                        channel_num = n2 // map_size
                        node = n2 - map_size*channel_num
                        
                        index = (input_indices[0,node] == n1).nonzero().item()
                        
                        self.remove_mask[cur_layer][channel_num, node, index] = 0
                
            elif l2_name == "pooling":
                print("pooling remove = ", len(remove_e))
                k = l2_dim["kernel"]
                s = l2_dim["stride"]
                
                tensor_2d = torch.arange(l1_nodes).reshape(1,l1_channel,l1_size,l1_size).float()
        
                indices = F.unfold(tensor_2d, (k,k), stride = s)
                indices = indices.view(1, l1_channel, k*k, -1)
                input_indices = indices.view(1*l1_channel, k*k, -1).int()
                
                map_size = l2_size**2
                
                if cur_layer not in self.remove_mask.keys():
                    self.remove_mask[cur_layer] = torch.ones((l2_channel, input_indices.shape[1], input_indices.shape[2]))
                
                if len(remove_e) > 0:
                    for e in remove_e:
                        n1 = e[0] - self.cur_total_nodes
                        n2 = e[1] - self.cur_total_nodes - l1_nodes
                        
                        channel_num = n2 // map_size
                        node = n2 - map_size*channel_num
                        index = (input_indices[channel_num,:,node] == n1).nonzero().item()
                        
                        self.remove_mask[cur_layer][channel_num, index, node] = 0
                        
            else:
                print("FC remove = ", len(remove_e))
                if cur_layer not in self.remove_mask.keys():
                    self.remove_mask[cur_layer] = torch.ones((l1_nodes, l2_nodes))
                
                if len(remove_e) > 0:
                    for e in remove_e:
                        n1 = e[0] - self.cur_total_nodes
                        n2 = e[1] - self.cur_total_nodes - l1_nodes

                        self.remove_mask[cur_layer][n1,n2] = 0
                        
            self.cur_total_nodes += l1_nodes
            cur_layer += 1
                
        # self.edge_set |= new_edge_set
    
    

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)
    
        
    # max Pooling
    def maxpooling(self, ori, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _ = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim, _ = self.get_layer_info(l2)
        
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
        
        k = l2_dim["kernel"]
        s = l2_dim["stride"]
        
        batch = ori.shape[0]
        
        i_unf = F.unfold(ori, (k, k), stride=s) 
        i_unf = i_unf.view(batch, l1_channel, k*k, -1)
        
        res = None
        for i in range(l2_channel):
            y = mask[i] * i_unf[:,i,:,:].unsqueeze(1)
            res = y if res == None else torch.cat((res, y), axis=1)
            
        res = res.view(batch*l1_channel, k*k, -1)
        
        out = torch.max(res, dim = 1, keepdim=True).values

        out = out.view(batch,l1_channel, l2_size, l2_size)

        return out


    def CNN(self, ori, kernel, b, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _ = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim, _ = self.get_layer_info(l2)
        
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
        
        s = l2_dim["stride"]
        
        w = kernel.view(kernel.size(0),-1).T
        
        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]), stride=s).transpose(1,2)
        
        res = None
        for i in range(l2_channel):
            y = (mask[i]*ori_unf.unsqueeze(1)) @ w[:,i]
            res = y if res == None else torch.cat((res, y), axis=1)
            
        y = F.fold(res, (l2_size,l2_size), (1,1))

        y = y + b[None,:,:,None]
        
        return y
    
    
    def linear(self, x, fc_layer, l1, l2):
        # print(self.cur_total_nodes)
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _ = self.get_layer_info(l1)
        
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
        
        with torch.no_grad():
            w = fc_layer.weight
            w *= mask.T  # Apply the transposed mask directly to w
            fc_layer.weight.copy_(w)
                
        y = fc_layer(x)
        
        return y
    
    
    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        self.cur_total_nodes = 0
        
        # first CNN
        x_cov1 = self.activation(self.CNN(x, self.conv1.weight, self.conv1.bias.unsqueeze(1), 1, 2))
        
        # # first pooloing
        # x_pool1 = self.maxpooling(x_cov1, 2, 3)
        
        # second CNN
        x_cov2 = self.activation(self.CNN(x_cov1, self.conv2.weight, self.conv2.bias.unsqueeze(1), 2, 3))
        
        # # second pooling
        # x_pool2 = self.maxpooling(x_cov2, 4, 5)
        
        # fc
        fc = x_cov2.view(-1, self.num_flat_features(x_cov2))
        
        fc1 = self.activation(self.linear(fc, self.fc1, 3, 4))
        
        fc2 = self.activation(self.linear(fc1, self.fc2, 4, 5))
        
        y = self.linear(fc2, self.fc3, 5, 6)
        
        return y



    
    # CNN using unfold/fold, calculate edges values
    def CNN_edges(self, ori, kernel, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _ = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim, _ = self.get_layer_info(l2)
        
        s = l2_dim["stride"]

        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]), stride=s).transpose(1,2)
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
        
        w = kernel.view(kernel.size(0),-1).T
        w = w.unsqueeze(0).transpose(2, 0)
        
        res = None
        for i in range(l2_channel):
            y = (mask[i]*ori_unf).transpose(1,2)
            edges = (y.unsqueeze(1) * w[None,i,:,:]).transpose(2,3)
            res = edges if res == None else torch.cat((res, edges), axis=1)
            
        dim = res.shape[1]*res.shape[2]*res.shape[3]
        edges_v = res.reshape(-1, dim) # batch * edge_num
        # edge_v = self.calculate_edge_v(res, w)

        return edges_v
    
    
    # Pooling
    def pooling_edges(self, input, l1, l2, k_size = 2, stride = 2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _ = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim, _ = self.get_layer_info(l2)
        
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
        
        #print(input.shape)
        batch = input.shape[0]
        channel = input.shape[1]

        i_unf = F.unfold(input, (k_size, k_size), stride=stride) 

        i_unf = i_unf.view(batch, channel, k_size*k_size, -1)
        
        res = None
        for i in range(l2_channel):
            y = mask[i] * i_unf[:,i,:,:].unsqueeze(1)
            res = y if res == None else torch.cat((res, y), axis=1)
        
        edges = res.transpose(2,3)

        dim = edges.shape[1]*edges.shape[2]*edges.shape[3]
        edges = edges.reshape(-1, dim) # batch * edge_num

        return edges
    
    
    # fullyconnected
    def fc_edges(self, x_tmp, layer, l1, l2):
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
            
        w = layer.weight.T * mask
        cur_shape = layer.weight.shape[0]*layer.weight.shape[1]
        
        edge_v = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)
        return edge_v


    # calculate edge weights
    def edge_w_batch(self, x):
        edge_value = None
        x_tmp = x
        
        nodes = x.view(-1, self.num_flat_features(x))
        
        # first CNN layer
        k1 = self.conv1.weight
        edge_v = (self.CNN_edges(x_tmp, k1, 1, 2)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.CNN(x, self.conv1.weight, self.conv1.bias.unsqueeze(1), 1, 2))
        
        x_tmp = x
        
        nodes = torch.cat((nodes, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        # # first max pooling
        # edge_v = (self.pooling_edges(x_tmp, 2, 3)).cpu().detach()
        # edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        # x = self.maxpooling(x, 2, 3)
        # x_tmp = x
        # nodes = torch.cat((nodes, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        # second CNN layer
        k2 = self.conv2.weight
        edge_v = (self.CNN_edges(x_tmp, k2, 2, 3)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.CNN(x, self.conv2.weight, self.conv2.bias.unsqueeze(1), 2, 3))
        nodes = torch.cat((nodes, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        # # second max pooling
        # edge_v = (self.pooling_edges(x_tmp, 4, 5)).cpu().detach()
        # edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        # x = self.maxpooling(x, 4, 5)

        # fully connected
        x = x.view(-1, self.num_flat_features(x)) # batch * input size
        x_tmp = x
        
        # fc1
        edge_v = (self.fc_edges(x_tmp, self.fc1, 3, 4)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.linear(x, self.fc1, 3, 4))
        x_tmp = x
        
        nodes = torch.cat((nodes, x), axis = 1)
        
        # fc2    
        edge_v = (self.fc_edges(x_tmp, self.fc2, 4, 5)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.linear(x, self.fc2, 4, 5))
        x_tmp = x
        
        nodes = torch.cat((nodes, x), axis = 1)

        # fc3
        edge_v = (self.fc_edges(x_tmp, self.fc3, 5, 6)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.fc3(x)
        nodes = torch.cat((nodes, x), axis = 1)
        
        return edge_value, nodes
    
    
    # calculate NN weights
    def get_weights(self, x1):
        edge_value = None
        x = torch.ones_like(x1)
        x_tmp = x
        
        # first CNN layer
        k1 = self.conv1.weight
        edge_v = (self.CNN_edges(x_tmp, k1, 1, 2)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.CNN(x, self.conv1.weight, self.conv1.bias.unsqueeze(1), 1, 2))
        x = torch.ones_like(x)
        x_tmp = x
        
        # # first max pooling
        # edge_v = (self.pooling_edges(x_tmp, 2, 3)).cpu().detach()
        # edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        # x = self.maxpooling(x, 2, 3)
        # x_tmp = x
        
        # second CNN layer
        k2 = self.conv2.weight
        edge_v = (self.CNN_edges(x_tmp, k2, 2, 3)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.CNN(x, self.conv2.weight, self.conv2.bias.unsqueeze(1), 2, 3))
        x = torch.ones_like(x)
        # # second max pooling
        # edge_v = (self.pooling_edges(x_tmp, 4, 5)).cpu().detach()
        # edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        # x = self.maxpooling(x, 4, 5)

        # fully connected
        x = x.view(-1, self.num_flat_features(x)) # batch * input size
        x_tmp = x
        
        # fc1
        edge_v = (self.fc_edges(x_tmp, self.fc1, 3, 4)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.linear(x, self.fc1, 3, 4))
        x = torch.ones_like(x)
        x_tmp = x
        
        # fc2    
        edge_v = (self.fc_edges(x_tmp, self.fc2, 4, 5)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        x = self.activation(self.linear(x, self.fc2, 4, 5))
        x = torch.ones_like(x)
        x_tmp = x
        
        # fc3
        edge_v = (self.fc_edges(x_tmp, self.fc3, 5, 6)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        return edge_value
    
    
    def get_new_edge_v(self, nodes_v, adjacent_m):
        adj_new = adjacent_m * nodes_v[0].view(1, -1)
        return adj_new



        
