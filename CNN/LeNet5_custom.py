import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class LeNet_custom(nn.Module):

    # network structure
    def __init__(self, model_info, edge_set):
        super(LeNet_custom, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
        self.model_info = model_info
        self.edge_set = edge_set
        self.cur_total_nodes = 0
        

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
        
        cur_channel = 1 if (cur_name == "fc") else cur_dim["channel"]
        
        if cur_name == "input":
            cur_nodes = cur_channel * cur_size**2
        else:
            cur_nodes = cur_size if (cur_name == "fc") else cur_dim["channel"]*(cur_size**2)
            
        return cur_nodes, cur_size, cur_channel, cur_dim
        
        
    
    # max Pooling
    def maxpooling(self, ori, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim = self.get_layer_info(l2)
        
        k = l2_dim["kernel"]
        s = l2_dim["stride"]
        
        batch = ori.shape[0]

        out_s0 = (ori.shape[2] - k) // s + 1
        out_s1 = (ori.shape[3] - k) // s + 1
        
        remove_e = [e for e in self.edge_set if (e[1] < (l2_nodes + l1_nodes + self.cur_total_nodes) and (e[1] >= l1_nodes + self.cur_total_nodes))]
        
        tensor_2d = torch.arange(l1_nodes).reshape(1,l1_channel,l1_size,l1_size).float()
        
        indices = F.unfold(tensor_2d, (k,k), stride = s)
        indices = indices.view(1, l1_channel, k*k, -1)
        input_indices = indices.view(l1_channel, k*k, -1).int()
        
        map_size = out_s0 * out_s1
        
        i_unf = F.unfold(ori, (k, k), stride=s) 
        i_unf = i_unf.view(batch, l1_channel, k*k, -1)
        i_unf = i_unf.view(batch*l1_channel, k*k, -1)
        
        print(f'Remove edge num {len(remove_e)}... \n')
        for e in remove_e:
            # print(f'current edge is from {e[0]} to {e[1]}... \n')
            n1 = e[0] - self.cur_total_nodes
            n2 = e[1] - self.cur_total_nodes - l1_nodes
            
            # print(f'current edge is from {n1} to {n2}... \n')
            
            channel_num = n2 // map_size
            node = n2 - map_size*channel_num
            # print(f'channel : {channel_num}, node {node}...\n')
            index = (input_indices[channel_num,:,node] == n1).nonzero().item()
            
            i_unf[channel_num::batch, index, node] = 0

        # claculate edge value
        edges = i_unf.view(batch, l1_channel, k*k, -1)
        edges = edges.transpose(2,3)

        dim = edges.shape[1]*edges.shape[2]*edges.shape[3]
        edges = edges.reshape(-1, dim) # batch * edge_num
        
        out = torch.max(i_unf, dim = 1, keepdim=True).values

        out = out.view(batch,l1_channel, out_s0, out_s1)
        
        self.cur_total_nodes += l1_nodes

        return out, edges


    def CNN(self, ori, kernel, b, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim = self.get_layer_info(l2)
        
        k = l2_dim["kernel"]
        s = l2_dim["stride"]
        batch = ori.shape[0]
        
        remove_e = [e for e in self.edge_set if (e[1] < (l2_nodes + l1_nodes + self.cur_total_nodes) and (e[1] >= l1_nodes + self.cur_total_nodes))]
        
        tensor_2d = torch.arange(l1_nodes).reshape(1,l1_channel,l1_size,l1_size).float()
        input_indices = F.unfold(tensor_2d, (k,k), stride = s).transpose(1,2).int()
      
        out_size = (ori.shape[2] - kernel.shape[2])//1 +1
        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3])).transpose(1,2)
        ori_repeat = ori_unf.repeat(l2_channel, 1, 1)
        
        assert(out_size == l2_size)
        map_size = out_size**2
        
        print(f'Remove edge num {len(remove_e)}... \n')
        for e in remove_e:
            n1 = e[0] - self.cur_total_nodes
            n2 = e[1] - self.cur_total_nodes - l1_nodes
            
            channel_num = n2 // map_size
            node = n2 - map_size*channel_num
            
            index = (input_indices[0,node] == n1).nonzero().item()
            
            ori_repeat[channel_num::batch, node, index] = 0
        
        w = kernel.view(kernel.size(0),-1).T
        w = w.unsqueeze(0).transpose(2, 0)
        
        out = (ori_repeat @ (kernel.view(kernel.size(0),-1).T)).transpose(1,2)
        
        # calculate edge values
        ori_re = ori_repeat.transpose(1,2)
        edges = (ori_re.unsqueeze(1) * w[None,:,:,:]).transpose(2,3)
        
        # convert dimension
        res = out[::batch]
        edge_res = edges[::batch]
        for i in range(1, batch, 1):
            # print(out1[i::step].shape)
            res = torch.cat((res, out[i::batch]), dim=2)
            edge_res = torch.cat((edge_res, edges[i::batch]), dim=3)
        
        y = res[0, 0].unsqueeze(0)
        edges_y = edge_res[0,0].unsqueeze(0)
        for i in range(1, res.shape[0]):
            y = torch.cat((y, res[i, i].unsqueeze(0)), dim = 0)
            edges_y = torch.cat((edges_y, edge_res[i, i].unsqueeze(0)), dim=0)

        y = y.reshape(batch*l2_channel, 1, map_size)
        # print(edges_y.shape)
        edges_y = edges_y.transpose(1,2).reshape(l2_channel*batch, (k**2)*l1_channel, map_size).transpose(1,2)
        e_res = edges_y[::batch]
        res = y[::batch]
        for i in range(1, batch, 1):
            res = torch.cat((res, y[i::batch]), dim=0)
            e_res = torch.cat((e_res, edges_y[i::batch]), dim=0)

        e_res = e_res.reshape(batch,l2_channel,k**2,l1_channel*map_size)
        dim = e_res.shape[1]*e_res.shape[2]*e_res.shape[3]
        edge_v = e_res.reshape(-1, dim) # batch * edge_num
        
        y = res.reshape(batch,l2_channel,map_size)

        y = F.fold(y, (out_size,out_size), (1,1))

        y = y + b[None,:,:,None]
        
        self.cur_total_nodes += l1_nodes

        return y, edge_v
    
    
    def linear(self, x, fc_layer, l1, l2):
        # print(self.cur_total_nodes)
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim = self.get_layer_info(l2)
        
        remove_e = [e for e in self.edge_set if (e[1] < (l2_nodes + l1_nodes + self.cur_total_nodes) and (e[1] >= l1_nodes + self.cur_total_nodes))]
        
        print(f'Remove edge num {len(remove_e)}... \n')
        for e in remove_e:
            n1 = e[0] - self.cur_total_nodes
            n2 = e[1] - self.cur_total_nodes - l1_nodes
            
            with torch.no_grad():
                fc_layer.weight.T[n1,n2] = 0
        
        y = fc_layer(x)
        self.cur_total_nodes += l1_nodes
        
        # calculate edge value
        w = fc_layer.weight.T
        cur_shape = fc_layer.weight.shape[0]*fc_layer.weight.shape[1]
        
        edge_v = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x], axis=0)
        return y, edge_v
    
    
    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        self.cur_total_nodes = 0
        edge_value = None
        
        # first CNN
        x_cov1, edge_v = self.CNN(x, self.conv1.weight, self.conv1.bias.unsqueeze(1), 1, 2)
        x_cov1 = F.relu(x_cov1)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        # first pooloing
        x_pool1, edge_v = self.maxpooling(x_cov1, 2, 3)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        # second CNN
        x_cov2, edge_v = self.CNN(x_pool1, self.conv2.weight, self.conv2.bias.unsqueeze(1), 3, 4)
        x_cov2 = F.relu(x_cov2)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        # second pooling
        x_pool2, edge_v = self.maxpooling(x_cov2, 4, 5)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        # fc
        fc = x_pool2.view(-1, self.num_flat_features(x_pool2))
        
        fc1, edge_v = self.linear(fc, self.fc1, 5, 6)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        fc1 = F.relu(fc1)
        
        fc2, edge_v = self.linear(fc1, self.fc2, 6, 7)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        fc2 = F.relu(fc2)
        
        y, edge_v = self.linear(fc2, self.fc3, 7, 8)
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        return y, edge_value




        
