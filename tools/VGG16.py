import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


class VGG16CustomTail(nn.Module):
    def __init__(self, model_info, edge_set, device, num_classes=1000, pretrained_path=None, start_l=1):
        super(VGG16CustomTail, self).__init__()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        
        self.activation = nn.ReLU(inplace=True)

        # Load VGG16 for copying weights
        base_model = models.vgg16()

        # Keep front part up to conv4_2
        self.features_early = nn.Sequential(*base_model.features[:24])  # Conv4_2 inclusive

        # Explicitly define conv4_3, conv5_1, conv5_2, conv5_3
        # self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.model_info = model_info
        self.edge_set = edge_set if edge_set != None else set()
        self.cur_total_nodes = 0
        self.device = device
        self.remove_mask = dict()
        self.__build_remove_mask__(self.edge_set, start_l)


        # Load pretrained weights
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            base_model.load_state_dict(state_dict)
            
            # print(base_model.features)

            # Copy conv layers
            self.features_early.load_state_dict(base_model.features[:24].state_dict())
            # print(type(base_model.features[23]))
            # self.conv4_3.load_state_dict(base_model.features[23].state_dict())
            self.conv5_1.load_state_dict(base_model.features[24].state_dict())
            self.conv5_2.load_state_dict(base_model.features[26].state_dict())
            self.conv5_3.load_state_dict(base_model.features[28].state_dict())

            # Copy FC layers
            self.fc1.load_state_dict(base_model.classifier[0].state_dict())
            self.fc2.load_state_dict(base_model.classifier[3].state_dict())
            self.fc3.load_state_dict(base_model.classifier[6].state_dict())

            print(f"Loaded pretrained VGG16 weights from {pretrained_path}")


   
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
    

    
    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.normalize(x)
        
        x = self.features_early(x)

        x = self.activation(self.conv5_1(x))
        x = self.activation(self.conv5_2(x))
        x = self.activation(self.conv5_3(x))
        x = self.pool5(x)

        fc = x.view(-1, self.num_flat_features(x))
        fc1 = self.activation(self.linear(fc, self.fc1, 6, 7))
        fc2 = self.activation(self.linear(fc1, self.fc2, 7, 8))  
        y = self.linear(fc2, self.fc3, 8, 9)
        return y
    
    
    # CNN using unfold/fold, calculate edges values
    def CNN_edges(self, ori, kernel, l1, l2):
        # get l1, l2 info
        l2_nodes, l2_size, l2_channel, l2_dim, _, l2_padding, l2_pool = self.get_layer_info(l2)
        
        s = l2_dim["stride"]
        p = l2_padding

        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]), stride=s, padding = p).transpose(1,2)
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
        
        x = self.features_early(x)
        # x = self.activation(self.conv4_3(x))
        x = self.activation(self.conv5_1(x))
        x = self.activation(self.conv5_2(x))
        x = self.activation(self.conv5_3(x))
        x = self.pool5(x)
        
        # only count last 3 layers
        edge_value = None
        weights = None
        nodes = x.view(-1, self.num_flat_features(x))

        # fully connected
        x = x.view(-1, self.num_flat_features(x)) # batch * input size
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # fc1
        edge_v = (self.fc_edges(x_tmp, self.fc1, 6, 7)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc1,  6, 7)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)

        x = self.activation(self.linear(x, self.fc1,  6, 7))
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        nodes = torch.cat((nodes, x), axis = 1)
        
        # fc2    
        edge_v = (self.fc_edges(x_tmp, self.fc2, 7, 8)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc2, 7, 8)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.activation(self.linear(x, self.fc2, 7, 8))
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        nodes = torch.cat((nodes, x), axis = 1)

        # fc3
        edge_v = (self.fc_edges(x_tmp, self.fc3, 8, 9)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc3, 8, 9)).cpu().detach()
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc3(x)
        nodes = torch.cat((nodes, x), axis = 1)
        
        return edge_value, nodes, weights
    
    
    
    # weights regularization 
    def normalization_weight_w4(self, nodes, weights, dims, model_dims):
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
                        
                        # Shape: (batch_size, fan-in)
                        node_slice = nodes[:, neighbors]
                        weight_slice = weights[:, in_edges]
                        
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
                node_slice = nodes[:, neighbors]
                weight_slice = weights[:, in_edges]

                # Prevent divide-by-zero
                weights_inv1[:, in_edges] = 1.0 / (torch.abs(weight_slice))
                weights_inv2[:, in_edges] = 1.0 / (torch.abs(node_slice))
            
                n += 1
        
        return weights_inv1, weights_inv2
