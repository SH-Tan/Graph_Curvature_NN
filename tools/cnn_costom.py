import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CNN_custom(nn.Module):

    # network structure
    def __init__(self, model_info, device, input_c = 1):
        super(CNN_custom, self).__init__()
        # Conv layers with batch norm
        self.conv1 = nn.Conv2d(3, 32, 6, padding = 0, stride=2)
        
        self.conv2 = nn.Conv2d(32, 16, 6, padding = 0, stride=2)
               
        # fully connected layer with batch norm
        self.fc1 = nn.Linear(16 * 5 * 5, 120)    
        self.fc2 = nn.Linear(120, 10)    

        self.activation = nn.ReLU()
        
        self.model_info = model_info
        self.device = device
        
    
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
        
        k = l2_dim["kernel"]
        s = l2_dim["stride"]
        
        batch = ori.shape[0]
        
        i_unf = F.unfold(ori, (k, k), stride=s) 
        i_unf = i_unf.view(batch, l1_channel, k*k, -1)
        
        res = None
        for i in range(l2_channel):
            y = i_unf[:,i,:,:].unsqueeze(1)
            res = y if res == None else torch.cat((res, y), axis=1)
            
        res = res.view(batch*l1_channel, k*k, -1)
        
        out = torch.max(res, dim = 1, keepdim=True).values

        out = out.view(batch,l1_channel, l2_size, l2_size)

        return out


    def CNN(self, ori, kernel, b, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _ = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim, _ = self.get_layer_info(l2)
        
        s = l2_dim["stride"]
        
        w = kernel.view(kernel.size(0),-1).T
        
        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]), stride=s).transpose(1,2)
        
        res = None
        for i in range(l2_channel):
            y = (ori_unf.unsqueeze(1)) @ w[:,i]
            res = y if res == None else torch.cat((res, y), axis=1)
            
        y = F.fold(res, (l2_size,l2_size), (1,1))

        y = y + b[None,:,:,None]
        
        return y
    
    
    def linear(self, x, fc_layer, l1, l2):
        with torch.no_grad():
            w = fc_layer.weight
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
        x_cov1 = self.activation((self.CNN(x, self.conv1.weight, self.conv1.bias.unsqueeze(1), 1, 2)))
        
        # second CNN
        x_cov2 = self.activation((self.CNN(x_cov1, self.conv2.weight, self.conv2.bias.unsqueeze(1), 2, 3)))
        
        # fc
        fc = x_cov2.view(-1, self.num_flat_features(x_cov2))
        
        fc1 = self.activation(self.linear(fc, self.fc1, 3, 4))
        
        y = self.linear(fc1, self.fc2, 4, 5)
        
        return y



    
    # CNN using unfold/fold, calculate edges values
    def CNN_edges(self, ori, kernel, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _ = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim, _ = self.get_layer_info(l2)
        
        s = l2_dim["stride"]

        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]), stride=s).transpose(1,2)

        w = kernel.view(kernel.size(0),-1).T
        w = w.unsqueeze(0).transpose(2, 0)
        
        res = None
        for i in range(l2_channel):
            y = (ori_unf).transpose(1,2)
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
  
        #print(input.shape)
        batch = input.shape[0]
        channel = input.shape[1]

        i_unf = F.unfold(input, (k_size, k_size), stride=stride) 

        i_unf = i_unf.view(batch, channel, k_size*k_size, -1)
        
        res = None
        for i in range(l2_channel):
            y = i_unf[:,i,:,:].unsqueeze(1)
            res = y if res == None else torch.cat((res, y), axis=1)
        
        edges = res.transpose(2,3)

        dim = edges.shape[1]*edges.shape[2]*edges.shape[3]
        edges = edges.reshape(-1, dim) # batch * edge_num

        return edges
    
    
    # fullyconnected
    def fc_edges(self, x_tmp, layer, l1, l2):
        w = layer.weight.T
        cur_shape = layer.weight.shape[0]*layer.weight.shape[1]
        
        edge_v = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)
        return edge_v


    # calculate edge weights
    def NN_info_batch(self, x):
        edge_value = None
        x_tmp = x

        weights = None
        ones_tmp = torch.ones_like(x)
        nodes = x.view(-1, self.num_flat_features(x))
        
        # first CNN layer
        k1 = self.conv1.weight
        edge_v = (self.CNN_edges(x_tmp, k1, 1, 2)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.CNN_edges(ones_tmp, k1, 1, 2)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.activation((self.CNN(x, self.conv1.weight, self.conv1.bias.unsqueeze(1), 1, 2)))
        
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        nodes = torch.cat((nodes, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        # second CNN layer
        k2 = self.conv2.weight
        edge_v = (self.CNN_edges(x_tmp, k2, 2, 3)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k2, 2, 3)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)

        x = self.activation((self.CNN(x, self.conv2.weight, self.conv2.bias.unsqueeze(1), 2, 3)))
        nodes = torch.cat((nodes, x.view(-1, self.num_flat_features(x))), axis = 1)

        # fully connected
        x = x.view(-1, self.num_flat_features(x)) # batch * input size
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # fc1
        edge_v = (self.fc_edges(x_tmp, self.fc1, 3, 4)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc1, 3, 4)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.activation(self.linear(x, self.fc1, 3, 4))
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        nodes = torch.cat((nodes, x), axis = 1)

        # fc2
        edge_v = (self.fc_edges(x_tmp, self.fc2, 4, 5)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc2, 4, 5)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc2(x)
        nodes = torch.cat((nodes, x), axis = 1)
        
        return edge_value, nodes, weights
    
    

    def normalization_weight(self, nodes, weights, dims, model_dims):
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(nodes.device)

        current_l = 1
        start_col = 0
        end_col = 0
        weights_inv = torch.zeros_like(weights)
        weights_new = torch.zeros_like(weights)
        
        n = dims[0]  # Start from the first node of the second layer

        while n < nodes_num:
            if n >= prefix_dims[current_l]:
                current_l += 1
                start_col = end_col
                
                end_col += (dims[current_l-2] * dims[current_l-1])
                neighbors = torch.arange(prefix_dims[current_l-2], prefix_dims[current_l-1])
                step = dims[current_l-1]
            
            layer = model_dims[current_l]
            prev_layer = model_dims[current_l - 1]
            
            # Extract layer details
            cur_name = layer["name"]
            cur_dim = layer["dim"]
            pre_dim = prev_layer["dim"]
            
            # Determine channels and dimensions
            cur_channel = 1 if cur_name == "fc" else cur_dim['channel']
            pre_channel = 1 if prev_layer["name"] == "fc" else pre_dim['channel']
            pre_nodes_num = dims[current_l - 2]
            
            if cur_name in ["cnn", "pooling"]:
                k = cur_dim['kernel']
                s = cur_dim['stride']
                in_size = pre_dim['out_size']
                
                # Generate indices for previous layer's nodes
                tensor_2d = torch.arange(pre_nodes_num, device=nodes.device).reshape(1, pre_channel, in_size, in_size).float()
                indices = F.unfold(tensor_2d, (k, k), stride=s).transpose(1, 2).int()
                step = k ** 2
                end_col = start_col + step * pre_channel
                
                # Process all channels and positions at once
                for c in range(cur_channel):
                    for l in range(indices.shape[1]):
                        neighbors = indices[0,l] + prefix_dims[current_l-2] 
                        in_edges = torch.arange(start_col, end_col, device=nodes.device)
                        
                        # Compute weights and normalization
                        w = nodes[:, neighbors] * weights[:, in_edges]
                        positive_w = torch.where(w > 0, w, 0)

                        pos_sum = positive_w.sum(dim=1, keepdim=True)
                        sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
                        
                        positive_s_i = torch.where(sum > 0)[0] # indices
                        
                        sub_pos_a = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
                                
                        mask = sub_pos_a > 0
                        values = (sub_pos_a * (sum[positive_s_i] / pos_sum[positive_s_i]))
                        
                        sub_pos_a_inv = torch.where(mask, 1./values, torch.tensor(0.))
                        sub_pos = torch.where(mask, values, torch.tensor(0.))
    
                        weights_new[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos
                        weights_inv[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv
                    
                        n += 1
                        start_col = end_col
                        end_col = start_col + step*pre_channel
                end_col = start_col
            
            elif cur_name == "fc":
                in_edges = torch.arange(start_col + (n - prefix_dims[current_l-1]), end_col, step)
            
                w = nodes[:, neighbors] * weights[:, in_edges] # batch * neighbors.size()
                
                positive_w = torch.where(w > 0, w, torch.zeros_like(w))
                pos_sum = positive_w.sum(dim=1, keepdim=True) # batch * 1
                
                sum = torch.sum(w, axis = 1, keepdim=True) # batch * 1
                
                positive_s_i = torch.where(sum > 0)[0] # indices
                
                sub_pos_a = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
                            
                mask = sub_pos_a > 0
                values = (sub_pos_a * (sum[positive_s_i] / pos_sum[positive_s_i]))
                
                sub_pos_a_inv = torch.where(mask, 1./values, torch.tensor(0.))
                sub_pos = torch.where(mask, values, torch.tensor(0.))

                weights_new[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos
                weights_inv[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv
                n += 1
        
        return weights_new, weights_inv