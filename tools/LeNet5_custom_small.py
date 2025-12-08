import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import tools.utils as utils



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
        self.softmax = nn.Softmax(dim=1)
        
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
    
    
    
    def __build_remove_mask__(self, new_edge_set, num = 100000, start_l = 1, mask = "global"):
        cur_layer = start_l
        self.cur_total_nodes = 0
        total_layers = len(self.model_info)
        
        remove_num = 0
        
        if mask == "global"  and len(new_edge_set) > num:
            new_edge_set = new_edge_set[:num]
        
        while(cur_layer < total_layers):
            if remove_num >= num:
                break
            # get l1, l2 info
            l1_nodes, l1_size, l1_channel, l1_dim, l1_name = self.get_layer_info(cur_layer)
            l2_nodes, l2_size, l2_channel, l2_dim, l2_name = self.get_layer_info(cur_layer+1)
            
            remove_e = [e for e in new_edge_set if (e[1] < (l2_nodes + l1_nodes + self.cur_total_nodes) and (e[1] >= l1_nodes + self.cur_total_nodes)) \
                and (e[0] >= self.cur_total_nodes and e[0] < (l1_nodes + self.cur_total_nodes))]

            if l2_name == "cnn":
                k = l2_dim["kernel"]
                s = l2_dim["stride"]
                
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
                        remove_num += 1

                        if remove_num >= num:
                            break
                if remove_num >= num:
                    break
                
            else:
                if cur_layer not in self.remove_mask.keys():
                    self.remove_mask[cur_layer] = torch.ones((l1_nodes, l2_nodes))
                
                if len(remove_e) > 0:
                    for e in remove_e:
                        n1 = e[0] - self.cur_total_nodes
                        n2 = e[1] - self.cur_total_nodes - l1_nodes

                        self.remove_mask[cur_layer][n1,n2] = 0
                        remove_num += 1

                        if remove_num >= num:
                            break
                if remove_num >= num:
                    break

            self.cur_total_nodes += l1_nodes
            cur_layer += 1
    
    

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
    
  
    def is_gaussian_torch(self, x, corr_thresh=0.99):
        """
        x: 1D torch.Tensor
        corr_thresh: threshold for QQ-correlation
        """
        x = x.flatten()

        # sort your data
        x_sorted = torch.sort(x).values

        # theoretical normal quantiles (same as scipy probplot with dist="norm")
        n = x_sorted.numel()
        p = torch.linspace(1/(n+1), n/(n+1), n)
        q_theoretical = torch.distributions.Normal(0, 1).icdf(p)

        # compute Pearson correlation between x_sorted and normal quantiles
        vx = x_sorted - x_sorted.mean()
        vq = q_theoretical - q_theoretical.mean()

        corr = (vx * vq).sum() / torch.sqrt((vx**2).sum() * (vq**2).sum())

        return corr.item() > corr_thresh
    
    
    def w_norm_std(self, w, alpha=10):

        w_mean = torch.mean(w)
        w_std = torch.std(w)
        w_norm = torch.abs((w)/(alpha*w_std))
        return w_norm
    
    
    def norm_w_minmax(self, w, alpha=1):
        # alpha = 0: keep raw weights
        # alpha = 1: full min-max normalization

        w_abs = torch.abs(w)
        min_vals = w_abs.min(dim=1, keepdim=True)[0]
        max_vals = w_abs.max(dim=1, keepdim=True)[0]
        w_minmax = ((w_abs - min_vals) / (max_vals - min_vals + 1e-6)) + 1e-6

        # Soft mixture
        return (1 - alpha) * w_abs + alpha * w_minmax
    
        
    def norm_gaussian(self, w):
        is_g = self.is_gaussian_torch(w)
        
        if is_g:
            w_norm = self.norm_w_minmax(w)
        else:
            w_norm = self.w_norm_std(w)
        return w_norm
    

    # calculate edge weights
    def NN_info_batch(self, x):
        edge_value = None
        x_tmp = x

        weights = None
        ones_tmp = torch.ones_like(x)
        nodes = x.view(-1, self.num_flat_features(x))
        
        # Take absolute values
        nodes_abs = torch.abs(nodes)

        # Compute min and max per sample (along nodes)
        min_vals = nodes_abs.min(dim=1, keepdim=True)[0]  # shape (batch_size, 1)
        max_vals = nodes_abs.max(dim=1, keepdim=True)[0]  # shape (batch_size, 1)

        # Normalize to [0, 1]
        nodes = (nodes_abs - min_vals) / (max_vals - min_vals)
        
        # first CNN layer
        k1 = self.conv1.weight
        edge_v = (self.CNN_edges(x_tmp, k1, 1, 2)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.CNN_edges(ones_tmp, k1, 1, 2)).cpu().detach()
        # ones = self.norm_w_minmax(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.activation(self.CNN(x, self.conv1.weight, self.conv1.bias.unsqueeze(1), 1, 2))
        
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        # nodes = torch.cat((nodes, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x_flat = x.view(-1, self.num_flat_features(x))

        # Take absolute value and normalize per sample
        x_abs = torch.abs(x_flat)
        min_vals = x_abs.min(dim=1, keepdim=True)[0]
        max_vals = x_abs.max(dim=1, keepdim=True)[0]
        x_norm = (x_abs - min_vals) / (max_vals - min_vals)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        # second CNN layer
        k2 = self.conv2.weight
        edge_v = (self.CNN_edges(x_tmp, k2, 2, 3)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k2, 2, 3)).cpu().detach()
        # ones = self.norm_w_minmax(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)

        x = self.activation(self.CNN(x, self.conv2.weight, self.conv2.bias.unsqueeze(1), 2, 3))
        # nodes = torch.cat((nodes, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x_flat = x.view(-1, self.num_flat_features(x))

        # Take absolute value and normalize per sample
        x_abs = torch.abs(x_flat)
        min_vals = x_abs.min(dim=1, keepdim=True)[0]
        max_vals = x_abs.max(dim=1, keepdim=True)[0]
        x_norm = (x_abs - min_vals) / (max_vals - min_vals)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        

        # fully connected
        x = x.view(-1, self.num_flat_features(x)) # batch * input size
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # fc1
        edge_v = (self.fc_edges(x_tmp, self.fc1, 3, 4)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc1, 3, 4)).cpu().detach()
        # ones = self.norm_w_minmax(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.activation(self.linear(x, self.fc1, 3, 4))
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x), axis = 1)
        
        x_flat = x

        # Take absolute value and normalize per sample
        x_abs = torch.abs(x_flat)
        min_vals = x_abs.min(dim=1, keepdim=True)[0]
        max_vals = x_abs.max(dim=1, keepdim=True)[0]
        x_norm = (x_abs - min_vals) / (max_vals - min_vals)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        # fc2    
        edge_v = (self.fc_edges(x_tmp, self.fc2, 4, 5)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc2, 4, 5)).cpu().detach()
        # ones = self.norm_w_minmax(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.activation(self.linear(x, self.fc2, 4, 5))
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x), axis = 1)
        
        x_flat = x

        # Take absolute value and normalize per sample
        x_abs = torch.abs(x_flat)
        min_vals = x_abs.min(dim=1, keepdim=True)[0]
        max_vals = x_abs.max(dim=1, keepdim=True)[0]
        x_norm = (x_abs - min_vals) / (max_vals - min_vals)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)

        # fc3
        edge_v = (self.fc_edges(x_tmp, self.fc3, 5, 6)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc3, 5, 6)).cpu().detach()
        # ones = self.norm_w_minmax(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc3(x)
        # x = self.softmax(x)
        # nodes = torch.cat((nodes, x), axis = 1)
        
        x_flat = x

        # Take absolute value and normalize per sample
        x_abs = x_flat
        min_vals = x_abs.min(dim=1, keepdim=True)[0]
        max_vals = x_abs.max(dim=1, keepdim=True)[0]
        x_norm = (x_abs - min_vals) / (max_vals - min_vals)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        return edge_value, nodes, weights, nodes
    
    
    def normalization_weight_w1(self, nodes, weights, dims, model_dims):
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(nodes.device)

        current_l = 1
        start_col = 0
        end_col = 0
        
        weights_inv1 = torch.zeros_like(weights)
        weights_inv2 = torch.zeros_like(weights)
        
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
                        
                        sub_pos_a1 = weights[positive_s_i][:, in_edges]   
                        sub_pos_a2 = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
                                
                        mask = sub_pos_a1 > 0
                        values1 = sub_pos_a1 * (sum[positive_s_i] / pos_sum[positive_s_i])
                        values2 = torch.abs(sub_pos_a2 * (sum[positive_s_i] / pos_sum[positive_s_i]))
                        
                        sub_pos_a_inv1 = torch.where(mask, 1./values1, torch.tensor(0.))
                        sub_pos_a_inv2 = torch.where(mask, 1./values2, torch.tensor(0.))
                        weights_inv1[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv1
                        weights_inv2[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv2
                    
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
                
                sub_pos_a1 = weights[positive_s_i][:, in_edges]   
                sub_pos_a2 = weights[positive_s_i][:, in_edges] * nodes[positive_s_i][:, neighbors]
                        
                mask = sub_pos_a1 > 0
                values1 = sub_pos_a1 * (sum[positive_s_i] / pos_sum[positive_s_i])
                values2 = torch.abs(sub_pos_a2 * (sum[positive_s_i] / pos_sum[positive_s_i]))
                
                sub_pos_a_inv1 = torch.where(mask, 1./values1, torch.tensor(0.))
                sub_pos_a_inv2 = torch.where(mask, 1./values2, torch.tensor(0.))
                weights_inv1[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv1
                weights_inv2[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv2
                n += 1
        
        return weights_inv1, weights_inv2
    

    def normalization_weight_w2(self, nodes, weights, dims, model_dims):
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(nodes.device)

        current_l = 1
        start_col = 0
        end_col = 0
        
        weights_inv = torch.zeros_like(weights)
        
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

                weights_inv[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv
                n += 1
        
        return weights_inv
    
    
    
    def normalization_weight_w3(self, nodes, weights, dims, model_dims):
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(nodes.device)

        current_l = 1
        start_col = 0
        end_col = 0
        
        weights_inv1 = torch.zeros_like(weights)
        weights_inv2 = torch.zeros_like(weights)
        
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
                        final_mask_pos = (sub_pos_a1 >= 0) & (sub_pos != 0)
                        final_mask_neg = (sub_neg_a1 <= 0) & (sub_neg != 0)

                        values_pos = torch.abs(sub_pos * (Sum[positive_s_i] / pos_sum[positive_s_i]))
                        values_neg = torch.abs(sub_neg * (Sum[negative_s_i] / neg_sum[negative_s_i]))
                        
                        sub_pos_a_inv = torch.where(final_mask_pos, 1./values_pos, torch.tensor(0.))
                        sub_neg_a_inv = torch.where(final_mask_neg, 1./values_neg, torch.tensor(0.))
                        
                        weights_inv1[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv
                        weights_inv1[torch.tensor(negative_s_i)[:,None], torch.tensor(in_edges)] = sub_neg_a_inv
                        
                        # w2
                        values_pos2 = torch.abs(sub_pos_a2 * (Sum[positive_s_i] / pos_sum[positive_s_i]))
                        values_neg2 = torch.abs(sub_neg_a2 * (Sum[negative_s_i] / neg_sum[negative_s_i]))
                        
                        sub_pos_a_inv2 = torch.where(final_mask_pos, 1./values_pos2, torch.tensor(0.))
                        sub_neg_a_inv2 = torch.where(final_mask_neg, 1./values_neg2, torch.tensor(0.))

                        weights_inv2[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv2
                        weights_inv2[torch.tensor(negative_s_i)[:,None], torch.tensor(in_edges)] = sub_neg_a_inv2
                    
                        n += 1
                        start_col = end_col
                        end_col = start_col + step*pre_channel
                end_col = start_col
            
            elif cur_name == "fc":
                in_edges = torch.arange(start_col + (n - prefix_dims[current_l-1]), end_col, step)
            
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
                final_mask_pos = (sub_pos_a1 >= 0) & (sub_pos != 0)
                final_mask_neg = (sub_neg_a1 <= 0) & (sub_neg != 0)
                        
                values_pos = torch.abs(sub_pos * (Sum[positive_s_i] / pos_sum[positive_s_i]))
                values_neg = torch.abs(sub_neg * (Sum[negative_s_i] / neg_sum[negative_s_i]))
                
                sub_pos_a_inv = torch.where(final_mask_pos, 1./values_pos, torch.tensor(0.))
                sub_neg_a_inv = torch.where(final_mask_neg, 1./values_neg, torch.tensor(0.))
                
                weights_inv1[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv
                weights_inv1[torch.tensor(negative_s_i)[:,None], torch.tensor(in_edges)] = sub_neg_a_inv
                
                # w2
                values_pos2 = torch.abs(sub_pos_a2 * (Sum[positive_s_i] / pos_sum[positive_s_i]))
                values_neg2 = torch.abs(sub_neg_a2 * (Sum[negative_s_i] / neg_sum[negative_s_i]))
                
                sub_pos_a_inv2 = torch.where(final_mask_pos, 1./values_pos2, torch.tensor(0.))
                sub_neg_a_inv2 = torch.where(final_mask_neg, 1./values_neg2, torch.tensor(0.))

                weights_inv2[torch.tensor(positive_s_i)[:,None], torch.tensor(in_edges)] = sub_pos_a_inv2
                weights_inv2[torch.tensor(negative_s_i)[:,None], torch.tensor(in_edges)] = sub_neg_a_inv2
            
                n += 1
        
        return weights_inv1, weights_inv2
    
    
    
    
    def normalization_weight_w4(self, nodes, weights, dims, model_dims, edge_dims, device='cuda'):
        """
        CNN/FC layer-wise normalization using adjacency reconstruction per layer.
        Computes weights_inv1 (1/|w|), weights_inv2 (1/|input nodes|), weights_inv3 (1/|output nodes|).
        """
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(nodes.device)

        current_l = 1
        start_col = 0
        end_col = 0
        
        weights_inv1 = 1./torch.abs(weights)
        weights_inv2 = torch.zeros_like(weights)
        
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
                        
                        # Shape: (batch_size, fan-in)
                        node_slice = torch.abs(nodes[:, neighbors])
                        
                        # min_vals = node_slice.min(dim=1, keepdim=True)[0]
                        # max_vals = node_slice.max(dim=1, keepdim=True)[0]
                        
                        # if max_vals > min_vals:
                        #     node_slice = (node_slice - min_vals) / (max_vals - min_vals)
                        # else:
                        #     node_slice = node_slice
                        
                        # nodes[:, neighbors] = node_slice

                        weights_inv2[:, in_edges] = 1.0/torch.abs(node_slice)
        
                        n += 1
                        start_col = end_col
                        end_col = start_col + step*pre_channel
                end_col = start_col
            
            elif cur_name == "fc":
                in_edges = torch.arange(start_col + (n - prefix_dims[current_l-1]), end_col, step)
            
                # Shape: (batch_size, fan-in)
                node_slice = torch.abs(nodes[:, neighbors])
                
                # min_vals = node_slice.min(dim=1, keepdim=True)[0]
                # max_vals = node_slice.max(dim=1, keepdim=True)[0]
                
                # node_slice = (node_slice - min_vals) / (max_vals - min_vals)
                
                # nodes[:, neighbors] = node_slice

                # Prevent divide-by-zero
                weights_inv2[:, in_edges] = 1.0/torch.abs(node_slice)
                    
                n += 1

        return weights_inv1, weights_inv2

        
    
    # weights regularization 
    def normalization_weight_w4_old(self, nodes, weights, dims, model_dims, edge_dims):
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(nodes.device)

        current_l = 1
        start_col = 0
        end_col = 0
        
        weights_inv1 = torch.zeros_like(weights)
        weights_inv2 = torch.zeros_like(weights)
        weights_inv3 = torch.zeros_like(weights)
        
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
                        # w = nodes[:, neighbors] * weights[:, in_edges]
                        
                        # Shape: (batch_size, fan-in)
                        node_slice = torch.abs(nodes[:, neighbors])
                        weight_slice = weights[:, in_edges]
                        
                        min_vals = node_slice.min(dim=1, keepdim=True)[0]
                        max_vals = node_slice.max(dim=1, keepdim=True)[0]
                        
                        node_slice = (node_slice - min_vals) / (max_vals - min_vals)
                        
                       # Prevent divide-by-zero
                        weights_inv1[:, in_edges] = 1.0 / (torch.abs(weight_slice))
                        weights_inv2[:, in_edges] = 1.0 / (torch.abs(node_slice))
                        
                        n += 1
                        start_col = end_col
                        end_col = start_col + step*pre_channel
                end_col = start_col
            
            elif cur_name == "fc":
                in_edges = torch.arange(start_col + (n - prefix_dims[current_l-1]), end_col, step)
            
                # Shape: (batch_size, fan-in)
                node_slice = torch.abs(nodes[:, neighbors])
                weight_slice = weights[:, in_edges]
                
                min_vals = node_slice.min(dim=1, keepdim=True)[0]
                max_vals = node_slice.max(dim=1, keepdim=True)[0]
                
                node_slice = (node_slice - min_vals) / (max_vals - min_vals)

                # Prevent divide-by-zero
                weights_inv1[:, in_edges] = 1.0 / (torch.abs(weight_slice))
                weights_inv2[:, in_edges] = 1.0 / (torch.abs(node_slice))
                    
                n += 1
                
        ### ---------- SECOND PASS: outgoing edges ----------
        n = 0  # restart from first node of input layer
        current_l = 1
        start_col = 0
        end_col = 0
        
        while n < prefix_dims[-2]:  # go through all nodes except final output layer
            if n >= prefix_dims[current_l - 1]:
                # move to next layer's edges
                start_col = end_col
                
                if current_l >= len(dims):
                    break
                
                end_col += dims[current_l - 1] * dims[current_l]
                current_l += 1
                out_neighbors = torch.arange(prefix_dims[current_l-1], prefix_dims[current_l], device=nodes.device)
      

            layer = model_dims[current_l]
            prev_layer = model_dims[current_l - 1]
            cur_name = layer["name"]
            cur_dim = layer["dim"]
            pre_dim = prev_layer["dim"]

            if cur_name in ["cnn", "pooling"]:
                k = cur_dim['kernel']
                s = cur_dim['stride']
                in_size = pre_dim['out_size']
                pre_channel = 1 if prev_layer["name"] == "fc" else pre_dim['channel']
                cur_channel = 1 if cur_name == "fc" else cur_dim['channel']
                
                tensor_2d = torch.arange(pre_dim['out_size']**2 * pre_channel,
                                        device=nodes.device).reshape(1, pre_channel, in_size, in_size).float()
                indices = F.unfold(tensor_2d, (k, k), stride=s).transpose(1, 2).int()

                step = k ** 2

                for c in range(cur_channel):
                    for l in range(indices.shape[1]):
                        start_col += step * pre_channel
                end_col = start_col
                
                n = prefix_dims[current_l - 1]

            elif cur_name == "fc":
                tmp = start_col + (n - prefix_dims[current_l - 2])*dims[current_l-1]
                out_edges = torch.arange(tmp, tmp+dims[current_l-1], 1, device=nodes.device)

                node_slice = torch.abs(nodes[:, out_neighbors])
                # --- Step 1: normalize to [0, 1] ---
                min_vals = node_slice.min(dim=1, keepdim=True)[0]
                max_vals = node_slice.max(dim=1, keepdim=True)[0]
                
                node_slice = (node_slice - min_vals) / (max_vals - min_vals)

                # --- Step 2: compute weights ---
                weights_inv3[:, out_edges] = 1.0 / (torch.abs(node_slice))

                n += 1

        
        return weights_inv1, weights_inv2, weights_inv3
    
    
    
    