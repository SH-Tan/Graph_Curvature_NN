import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

class VGG16_CIFAR10(nn.Module):
    def __init__(self, model_info, edge_set, device, input_c = 3, num_classes = 10, start_l=1):
        super().__init__()

        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                              std=(0.2023, 0.1994, 0.2010))
        self.activation = nn.Tanh()

        # Helper for conv blocks (used for first 10 conv layers)
        def conv_block(in_c, out_c, num_convs):
            layers = []
            for _ in range(num_convs):
                layers += [
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.Tanh()
                ]
                in_c = out_c
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return layers

        # First 10 convolutional layers using conv_block (up to conv4_2)
        self.features = nn.Sequential(
            *conv_block(input_c, 64, 2),     # conv1_1, conv1_2
            *conv_block(64, 128, 2),         # conv2_1, conv2_2
            *conv_block(128, 256, 3),        # conv3_1, conv3_2, conv3_3
            *conv_block(256, 512, 3),        # conv4_1, conv4_2
        )

        # Last 3 conv layers (conv4_3, conv5_1, conv5_2) defined explicitly
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.bn5_3 = nn.BatchNorm2d(512)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 → 1x1

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)


        self.model_info = model_info
        self.edge_set = edge_set if edge_set != None else set()
        self.cur_total_nodes = 0
        self.device = device
        self.remove_mask = dict()
        self.__build_remove_mask__(self.edge_set, start_l)


    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)
    

    def get_layer_info(self, l):
        cur_name = self.model_info[l]["name"]
        cur_dim = self.model_info[l]["dim"]
        cur_size = cur_dim['out_size']
        cur_padding = cur_dim['padding'] if cur_name == "cnn" else 0
        cur_pool = cur_dim.get('pool', False)
        
        cur_channel = 1 if (cur_name == "fc") else cur_dim["channel"]
        
        if cur_name == "input":
            cur_nodes = cur_channel * cur_size**2
        else:
            cur_nodes = cur_size if (cur_name == "fc") else cur_dim["channel"]*(cur_size**2)
            
        if cur_pool:
            cur_size *= 2

        return cur_nodes, cur_size, cur_channel, cur_dim, cur_name, cur_padding, cur_pool
    
    
    def __build_remove_mask__(self, new_edge_set = set(), num = 100000, start_l = 1):
        cur_layer = start_l
        self.cur_total_nodes = 0
        remove_num = 0
        
        while(cur_layer < len(self.model_info)):
            if remove_num >= num:
                break
            # get l1, l2 info
            l1_nodes, l1_size, l1_channel, l1_dim, l1_name, l1_padding, l1_pool = self.get_layer_info(cur_layer)
            l2_nodes, l2_size, l2_channel, l2_dim, l2_name, l2_padding, l2_pool = self.get_layer_info(cur_layer+1)
            
            remove_e = [e for e in new_edge_set if (e[1] < (l2_nodes + l1_nodes + self.cur_total_nodes) and (e[1] >= l1_nodes + self.cur_total_nodes)) \
                and (e[0] >= self.cur_total_nodes and e[0] < (l1_nodes + self.cur_total_nodes))]
            

            if l2_name == "cnn":
                k = l2_dim["kernel"]
                s = l2_dim["stride"]
                if l1_pool:
                    l1_size = (int)(l1_size/2)
                    
                tensor_2d = torch.arange(l1_nodes).reshape(1,l1_channel,l1_size,l1_size).float()
                input_indices = F.unfold(tensor_2d, (k,k), stride = s, padding = l2_padding).transpose(1,2).int()
                
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
                

    def CNN(self, ori, kernel, b, l1, l2):
        # get l1, l2 info
        l2_nodes, l2_size, l2_channel, l2_dim, _, l2_padding, l2_pool = self.get_layer_info(l2)
        
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
        
        s = l2_dim["stride"]
        
        w = kernel.view(kernel.size(0),-1).T
        
        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]), stride=s, padding = l2_padding).transpose(1,2)
        
        res = None
        for i in range(l2_channel):
            y = (mask[i]*ori_unf.unsqueeze(1)) @ w[:,i]
            res = y if res == None else torch.cat((res, y), axis=1)
            
        y = F.fold(res, (l2_size,l2_size), (1,1))

        y = y + b[None,:,:,None]
        
        return y
    
    
    def linear(self, x, fc_layer, l1, l2):
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
        
        with torch.no_grad():
            w = fc_layer.weight
            w *= mask.T  # Apply the transposed mask directly to w
            fc_layer.weight.copy_(w)
                
        y = fc_layer(x)
        
        return y
    
    
    # def forward(self, x):
    #     x = self.normalize(x)
    #     x = self.features(x)
    #     x = self.activation(self.bn5_1(self.conv5_1(x)))
    #     x = self.activation(self.bn5_2(self.conv5_2(x)))
    #     x = self.pool5(x)
    #     x = self.activation(self.bn5_3(self.conv5_3(x)))
    #     x = x.view(x.size(0), -1)  # flatten
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.fc3(x)
    #     return x


    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.normalize(x)

        x = self.features(x)  # First 11 conv layers (conv1_1 to conv4_3)

        x = self.activation(self.bn5_1(self.conv5_1(x)))
        x = self.activation(self.bn5_2(self.conv5_2(x)))
        x = self.pool5(x)
        
        # third CNN
        # x_cov3 = self.activation(self.bn5_3(self.conv5_3(x)))
        x_cov3 = self.activation(self.bn5_3(self.CNN(x, self.conv5_3.weight, self.conv5_3.bias.unsqueeze(1), 7, 8)))
        
        # fc
        fc = x_cov3.view(-1, self.num_flat_features(x_cov3))
        
        fc1 = self.activation(self.linear(fc, self.fc1, 8, 9))
        
        fc2 = self.activation(self.linear(fc1, self.fc2, 9, 10))
        
        y = self.linear(fc2, self.fc3, 10, 11)
        
        return y


    # CNN using unfold/fold, calculate edges values
    def CNN_edges(self, ori, kernel, l1, l2):
        # get l1, l2 info
        l2_nodes, l2_size, l2_channel, l2_dim, _, l2_padding, l2_pool = self.get_layer_info(l2)
        
        s = l2_dim["stride"]
        p = l2_padding
        
        # Pad with value=1 (1 pixel on all 4 sides)
        # x_padded = F.pad(ori, pad=(p, p, p, p), mode='constant', value=1)

        # Now apply unfold without padding
        ori_unf = F.unfold(ori, (kernel.shape[2],kernel.shape[3]), stride=s, padding=p).transpose(1,2)

        # ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]), stride=s, padding = p).transpose(1,2)
        
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

        return edges_v
    

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
        x = self.normalize(x)
        x = self.features(x)  # First 11 conv layers (conv1_1 to conv4_3)

        x_cov1 = self.activation(self.bn5_1(self.CNN(x, self.conv5_1.weight, self.conv5_1.bias.unsqueeze(1), 5, 6)))
 
        x_cov2 = self.activation(self.bn5_2(self.CNN(x_cov1, self.conv5_2.weight, self.conv5_2.bias.unsqueeze(1), 6, 7)))
        x_cov2 = self.pool5(x_cov2)

        # only count last 4 layers
        edge_value = None
        weights = None
        x_tmp = x_cov2
        ones_tmp = torch.ones_like(x_cov2)
        nodes = x_cov2.view(-1, self.num_flat_features(x_cov2))
        
        # 13th CNN layer
        k1 = self.conv5_3.weight
        edge_v = (self.CNN_edges(x_tmp, k1, 7, 8)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.CNN_edges(ones_tmp, k1, 7, 8)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x_cov3 = self.activation(self.bn5_3(self.CNN(x_cov2, self.conv5_3.weight, self.conv5_3.bias.unsqueeze(1), 7, 8)))

        x_tmp = x_cov3
        ones_tmp = torch.ones_like(x_cov3)
        
        nodes = torch.cat((nodes, x_cov3.view(-1, self.num_flat_features(x_cov3))), axis = 1)

        # fully connected
        x = x_cov3.view(-1, self.num_flat_features(x_cov3)) # batch * input size
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # fc1
        edge_v = (self.fc_edges(x_tmp, self.fc1, 8, 9)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc1, 8, 9)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)

        x = self.activation(self.linear(x, self.fc1, 8, 9))
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        nodes = torch.cat((nodes, x), axis = 1)
        
        # fc2    
        edge_v = (self.fc_edges(x_tmp, self.fc2, 9, 10)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc2, 9, 10)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.activation(self.linear(x, self.fc2, 9, 10))
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        nodes = torch.cat((nodes, x), axis = 1)

        # fc3
        edge_v = (self.fc_edges(x_tmp, self.fc3, 10, 11)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc3, 10, 11)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc3(x)
        nodes = torch.cat((nodes, x), axis = 1)
        
        return edge_value, nodes, weights
    

    def normalization_weight_w3(self, nodes, weights, dims, model_dims):
        device = nodes.device
        nodes_num = nodes.shape[1]
        prefix_dims = torch.cumsum(torch.tensor(dims), dim=0)
        prefix_dims = torch.cat([torch.tensor([0]), prefix_dims]).to(device)

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
                p = cur_dim['padding']
                in_size = pre_dim['out_size']
                
                # Generate indices for previous layer's nodes
                tensor_2d = torch.arange(pre_nodes_num, device=nodes.device).reshape(1, pre_channel, in_size, in_size).float()
                indices = F.unfold(tensor_2d, (k, k), stride=s, padding = p).transpose(1, 2).int()
                step = k ** 2
                end_col = start_col + step * pre_channel
                
                # Process all channels and positions at once
                for c in range(cur_channel):
                    for l in range(indices.shape[1]):
                        neighbors = indices[0,l] + prefix_dims[current_l-2] 
                        in_edges = torch.arange(start_col, end_col, device=device)
                        
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
                        # mask_pos = sub_pos_a1 >= 0
                        # mask_neg = sub_neg_a1 <= 0
                        # mask_pos_w = sub_pos != 0
                        # mask_neg_w = sub_neg != 0
                        
                        final_mask_pos = (sub_pos_a1 >= 0) & (sub_pos != 0)
                        final_mask_neg = (sub_neg_a1 <= 0) & (sub_neg != 0)

                        values_pos = torch.abs(sub_pos * (Sum[positive_s_i] / pos_sum[positive_s_i]))
                        values_neg = torch.abs(sub_neg * (Sum[negative_s_i] / neg_sum[negative_s_i]))
                        
                        sub_pos_a_inv = torch.where(final_mask_pos, 1./values_pos, torch.tensor(0.))
                        sub_neg_a_inv = torch.where(final_mask_neg, 1./values_neg, torch.tensor(0.))
                        
                        # sub_pos_a_inv = torch.where(mask_pos_w, sub_pos_a_inv, torch.tensor(0.))
                        # sub_neg_a_inv = torch.where(mask_neg_w, sub_neg_a_inv, torch.tensor(0.))

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
                # mask_pos = sub_pos_a1 >= 0
                # mask_neg = sub_neg_a1 <= 0
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