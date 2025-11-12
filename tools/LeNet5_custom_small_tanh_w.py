import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .q_exponential import q_exponential



class LeNet_custom_v2(nn.Module):

    # network structure
    def __init__(self, model_info, edge_set, device, prefix_dims, input_c = 1):
        super(LeNet_custom_v2, self).__init__()
        self.conv1 = nn.Conv2d(input_c, 6, 6, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 6, stride=2)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        self.model_info = model_info
        self.edge_set = edge_set if edge_set != None else set()
        self.cur_total_nodes = 0
        self.device = device
        self.prefix_dims = prefix_dims
        self.remove_mask = dict()
        self.init_remove_mask()
        
    
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
    
    
    def init_remove_mask(self):
        total_layers = len(self.model_info)

        # === 1. Prebuild all-one masks for every layer ===
        self.remove_mask = {}
        for l in range(1, total_layers):  # skip input layer
            layer_info = self.model_info[l+1]
            l1_info = self.model_info[l]

            if layer_info["name"] == "cnn":
                out_ch = layer_info["dim"]["channel"]
                k = layer_info["dim"]["kernel"]
                in_ch = l1_info["dim"]["channel"]
                self.remove_mask[l] = torch.ones((out_ch, in_ch, k, k))
            else:
                l1_nodes = l1_info["dim"]["out_size"]
                l2_nodes = layer_info["dim"]["out_size"]
                
                if l1_info["name"] == "cnn":
                    l1_nodes = l1_nodes**2 * l1_info["dim"]["channel"]
                
                self.remove_mask[l] = torch.ones((l1_nodes, l2_nodes))
    
    
    
    def __build_remove_mask__(
        self,
        mixed_set,
        num=100000,
        mask="global"
    ):
        total_layers = len(self.model_info)
        remove_num = 0

        # Global truncation
        if mask == "global" and len(mixed_set) > num:
            mixed_set = mixed_set[:num]

        # === 3. Main removal loop ===
        for item in mixed_set:
            item_type = item[0]

            # --- CNN weight-level removal ---
            if item_type == "weight":
                w_idx = item[1]
                if len(w_idx) != 5:
                    print(w_idx)
                    continue
                layer, oc, ic, kh, kw = w_idx
                if (layer+1) not in self.remove_mask or self.model_info[layer+2]["name"] != "cnn":
                    print(f"no cnn mask: {layer}")
                    continue

                mask_tensor = self.remove_mask[layer+1]
                if (
                    0 <= oc < mask_tensor.shape[0]
                    and 0 <= ic < mask_tensor.shape[1]
                    and 0 <= kh < mask_tensor.shape[2]
                    and 0 <= kw < mask_tensor.shape[3]
                ):
                    mask_tensor[oc, ic, kh, kw] = 0
                    remove_num += 1
                else:
                    print(w_idx, mask_tensor.shape)
                continue

            # --- FC edge-level removal (using prefix_dims) ---
            elif item_type == "edge":
                i, j = item[1], item[2]
                # Determine layer index using prefix_dims
                l1 = np.searchsorted(self.prefix_dims, i, side="right") - 1
                l2 = np.searchsorted(self.prefix_dims, j, side="right") - 1

                # Ensure it’s a valid FC layer
                if (
                    l1 >= 0
                    and l2 > l1
                    and l2 < total_layers
                    and self.model_info[l2+1]["name"] != "cnn"
                ):
                    local_i = i - self.prefix_dims[l1]
                    local_j = j - self.prefix_dims[l2]
                    mask_tensor = self.remove_mask[l2]
                    if (
                        0 <= local_i < mask_tensor.shape[0]
                        and 0 <= local_j < mask_tensor.shape[1]
                    ):
                        mask_tensor[local_i, local_j] = 0
                        remove_num += 1
                    else:
                        print(i, j, l1, l2, mask_tensor.shape)
                else:
                    print(i, j, l1, l2, self.model_info[l2]["name"])
                continue
            else:
                print(item_type)

        print(f"Removed {remove_num} total connections/weights.")
    
    

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
        
        # load mask
        mask = self.remove_mask[l1].to(self.device)  # shape: [out_ch, in_ch, k, k]

        s = l2_dim["stride"]
        kH, kW = kernel.shape[2], kernel.shape[3]
        
        # Flatten kernel and mask
        w = kernel.view(l2_channel, -1)  # [out_ch, in_ch*k*k]
        mask_flat = mask.view(l2_channel, -1)  # same shape

        # Masked kernel
        w_masked = w * mask_flat

        # Unfold input
        ori_unf = F.unfold(ori, (kH, kW), stride=s)  # [B, in_ch*k*k, H_out*W_out]
        ori_unf = ori_unf.transpose(1, 2)  # [B, H_out*W_out, in_ch*k*k]

        # Batch matrix multiply: [B, H_out*W_out, in_ch*k*k] @ [in_ch*k*k, out_ch] -> [B, H_out*W_out, out_ch]
        res = torch.matmul(ori_unf, w_masked.T)

        # Fold back
        y = F.fold(res.transpose(1, 2), (l2_size, l2_size), (1, 1))  # [B, out_ch, H_out, W_out]

        # Add bias
        y = y + b[None, :, :, None]
        
        return y
    
    
    def linear(self, x, fc_layer, l1, l2):
        
        mask = self.remove_mask[l1].to(self.device) 
        
        # Apply mask without modifying original weights permanently
        w_masked = fc_layer.weight * mask.T  # [out_features, in_features]

        # Perform linear manually
        y = F.linear(x, w_masked, fc_layer.bias)
        
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
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc3(x)
        x = self.softmax(x)
        # nodes = torch.cat((nodes, x), axis = 1)
        
        x_flat = x

        # Take absolute value and normalize per sample
        x_abs = torch.abs(x_flat)
        min_vals = x_abs.min(dim=1, keepdim=True)[0]
        max_vals = x_abs.max(dim=1, keepdim=True)[0]
        x_norm = (x_abs - min_vals) / (max_vals - min_vals)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        
        return edge_value, nodes, weights
    
    
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

        
    
    
    
    
    def normalization_weight_w6(self, nodes, weights, dims, model_dims, q):
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(nodes.device)

        current_l = 1
        start_col = 0
        end_col = 0
        
        q_exp = q_exponential(q)
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
                        in_edges = torch.arange(start_col, end_col, device=nodes.device)
                        
                        w = weights[:, in_edges]
                        mask = (w != 0)
                        if nodes[:, n].item() <= 0:
                            weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] = 1.0/abs(q_exp.q_exponential_series(-w))
                        else:
                            weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] = 1.0/abs(q_exp.q_exponential_series(w))
                        
                        weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] *= mask
                        
                        n += 1
                        start_col = end_col
                        end_col = start_col + step*pre_channel
                end_col = start_col
            
            elif cur_name == "fc":
                in_edges = torch.arange(start_col + (n - prefix_dims[current_l-1]), end_col, step)
            
                w = weights[:, in_edges]
                mask = (w != 0)
                if nodes[:, n].item() <= 0:
                    weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] = 1.0/abs(q_exp.q_exponential_series(-w))
                else:
                    weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] = 1.0/abs(q_exp.q_exponential_series(w))
                
                weights_inv[torch.arange(weights_inv.shape[0])[:, None], torch.tensor(in_edges)] *= mask
                n += 1
        
        return weights_inv
    