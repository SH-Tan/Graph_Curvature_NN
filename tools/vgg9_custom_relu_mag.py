import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np



class VGG9_CIFAR10(nn.Module):
    def __init__(self, model_info, edge_set, device, prefix_dims = [], input_c = 3, num_classes = 10, start_l=1):
        super().__init__()

        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                              std=(0.2023, 0.1994, 0.2010))


        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
        self.conv1_1 = nn.Conv2d(input_c, 64, kernel_size=2, padding=0, stride=2)  # 16
        self.bn1_1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=0, stride = 1)  # 14
        self.bn2_1 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, padding=0)  # 12
        self.bn3_1 = nn.BatchNorm2d(128)
        
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2)  # 5
        self.bn3_2 = nn.BatchNorm2d(128)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=0, stride = 1)  # 3
        self.bn4_1 = nn.BatchNorm2d(256)

        # Last 3 conv layers (conv4_3, conv5_1, conv5_2) defined explicitly
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=0, stride = 1)  # 1
        self.bn4_2 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.prefix_dims = prefix_dims
        self.model_info = model_info
        self.edge_set = edge_set if edge_set != None else list()
        self.cur_total_nodes = 0
        self.device = device
        self.remove_mask = {}


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
    
    
    @torch.no_grad()
    def backup_frozen_params(self):
        self._frozen_backup = {}

        for l, layer in [
            (1, self.conv1_1), (2, self.conv2_1),
            (3, self.conv3_1), (4, self.conv3_2),
            (5, self.conv4_1), (6, self.conv4_2),
            (7, self.fc1), (8, self.fc2), (9, self.fc3)
        ]:
            mask = self.remove_mask[l].to(self.device)

            self._frozen_backup[l] = {
                "weight": layer.weight.detach().clone(),
                "bias": None if layer.bias is None else layer.bias.detach().clone()
            }
            
    
    @torch.no_grad()
    def zero_frozen_params(self):
        for l, layer in [
            (1, self.conv1_1), (2, self.conv2_1),
            (3, self.conv3_1), (4, self.conv3_2),
            (5, self.conv4_1), (6, self.conv4_2),
            (7, self.fc1), (8, self.fc2), (9, self.fc3)
        ]:
            mask = self.remove_mask[l].to(self.device)

            layer.weight[mask == 1] = 0.0

            if layer.bias is not None:
                bias_mask = mask.view(mask.size(0), -1).all(dim=1)
                layer.bias[bias_mask] = 0.0


    @torch.no_grad()
    def restore_frozen_params(self):
        for l, layer in [
            (1, self.conv1_1), (2, self.conv2_1),
            (3, self.conv3_1), (4, self.conv3_2),
            (5, self.conv4_1), (6, self.conv4_2),
            (7, self.fc1), (8, self.fc2), (9, self.fc3)
        ]:
            if l not in self._frozen_backup:
                continue

            layer.weight.copy_(self._frozen_backup[l]["weight"])

            if layer.bias is not None:
                layer.bias.copy_(self._frozen_backup[l]["bias"])

    
        
    @torch.no_grad()
    def build_global_magnitude_remove_mask(
        self,
        ratio=0.5,
        freeze_smallest=True
    ):
        """
        Build self.remove_mask using GLOBAL weight magnitude.
        
        mask == 0 → freeze
        mask == 1 → train
        """

        self.remove_mask = {}

        layer_map = {
            1: self.conv1_1,
            2: self.conv2_1,
            3: self.conv3_1,
            4: self.conv3_2,
            5: self.conv4_1,
            6: self.conv4_2,
            7: self.fc1,
            8: self.fc2,
            9: self.fc3,
        }

        # --------------------------------------------------
        # 1. collect ALL weights globally
        # --------------------------------------------------
        all_weights = []
        for layer in layer_map.values():
            all_weights.append(layer.weight.data.view(-1))

        all_weights = torch.cat(all_weights)
        all_abs = all_weights.abs()

        k = int(ratio * all_abs.numel())
        if k == 0:
            for l, layer in layer_map.items():
                self.remove_mask[l] = torch.ones_like(layer.weight)
            return

        # --------------------------------------------------
        # 2. global threshold
        # --------------------------------------------------
        if freeze_smallest:
            thresh = torch.topk(all_abs, k, largest=False).values.max()
            freeze_fn = lambda W: (W.abs() <= thresh)
        else:
            thresh = torch.topk(all_abs, k, largest=True).values.min()
            freeze_fn = lambda W: (W.abs() >= thresh)

        # --------------------------------------------------
        # 3. build per-layer masks
        # --------------------------------------------------
        for l, layer in layer_map.items():
            W = layer.weight.data
            freeze_mask = freeze_fn(W)

            # mask == 0 → freeze, mask == 1 → train
            self.remove_mask[l] = (~freeze_mask).float().to(self.device)


        
        
    def register_freeze_grad(self, freeze_on_mask_one=True):
        self.register_conv_mask(self.conv1_1, 1, freeze_on_mask_one)
        self.register_conv_mask(self.conv2_1, 2, freeze_on_mask_one)
        self.register_conv_mask(self.conv3_1, 3, freeze_on_mask_one)
        self.register_conv_mask(self.conv3_2, 4, freeze_on_mask_one)
        self.register_conv_mask(self.conv4_1, 5, freeze_on_mask_one)
        self.register_conv_mask(self.conv4_2, 6, freeze_on_mask_one)

        self.register_linear_mask(self.fc1, 7, freeze_on_mask_one)
        self.register_linear_mask(self.fc2, 8, freeze_on_mask_one)
        self.register_linear_mask(self.fc3, 9, freeze_on_mask_one)



    def freeze_grad_hook(self, mask, freeze_on_mask_one=True):
        """
        Returns a gradient hook that freezes parameters according to mask.
        """
        mask = mask.float()

        if freeze_on_mask_one:
            # 1 → freeze → grad = 0
            grad_multiplier = 1.0 - mask
        else:
            # 0 → freeze → grad = 0
            grad_multiplier = mask

        def hook(grad):
            return grad * grad_multiplier

        return hook
        
        
    def register_conv_mask(self, conv_layer, l1, freeze_on_mask_one=False):
        mask = self.remove_mask[l1].to(self.device)  # [out, in, kH, kW]
        mask = mask.float()

        # ----------------------
        # weight
        # ----------------------
        def weight_grad_hook(grad):
            return grad * mask

        conv_layer.weight.register_hook(weight_grad_hook)

        # ----------------------
        # bias
        # ----------------------
        if conv_layer.bias is not None:
            # if ANY weight to output channel is trainable → bias trainable
            bias_mask = (mask.view(mask.size(0), -1).sum(dim=1) > 0).float()

            def bias_grad_hook(grad):
                return grad * bias_mask

            conv_layer.bias.register_hook(bias_grad_hook)

            
            
    def register_linear_mask(self, fc_layer, l1, freeze_on_mask_one=False):
        mask = self.remove_mask[l1].to(self.device)  # shape [out, in]
        mask = mask.float()

        # ----------------------
        # weight
        # ----------------------
        def weight_grad_hook(grad):
            return grad * mask

        fc_layer.weight.register_hook(weight_grad_hook)

        # ----------------------
        # bias
        # ----------------------
        if fc_layer.bias is not None:
            # freeze bias if ALL incoming weights are frozen
            # mask == 1 → train, 0 → freeze
            bias_mask = (mask.sum(dim=1) > 0).float()  # shape [out]

            def bias_grad_hook(grad):
                return grad * bias_mask

            fc_layer.bias.register_hook(bias_grad_hook)





    def CNN(self, ori, kernel, b, l1, l2):
        # get l1, l2 info
        l1_nodes, l1_size, l1_channel, l1_dim, _, _, _ = self.get_layer_info(l1)
        l2_nodes, l2_size, l2_channel, l2_dim, _, _, _ = self.get_layer_info(l2)
        
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
        x = self.normalize(x)
        
        x = self.activation(self.bn1_1(self.conv1_1(x)))
        
        # x = self.activation(self.bn1_2(self.conv1_2(x)))
        
        x = self.activation(self.bn2_1(self.conv2_1(x)))
        
        # x = self.activation(self.bn2_2(self.conv2_2(x)))
        
        x = self.activation(self.bn3_1(self.conv3_1(x)))
        
        x = self.activation(self.bn3_2(self.conv3_2(x)))
        
        # x = self.activation(self.bn3_3(self.conv3_3(x)))
        
        x = self.activation(self.bn4_1(self.conv4_1(x)))
        
        x = self.activation(self.bn4_2(self.conv4_2(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        x = self.fc3(x)
        return x


    def forward1(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.normalize(x)

         # first CNN
        x_cov1 = self.activation(self.bn1_1(self.CNN(x, self.conv1_1.weight, self.conv1_1.bias.unsqueeze(1), 1,2)))
        # x_cov2 = self.activation(self.bn1_2(self.CNN(x_cov1, self.conv1_2.weight, self.conv1_2.bias.unsqueeze(1), 2,3)))

         # second CNN
        x_cov2 = self.activation(self.bn2_1(self.CNN(x_cov1, self.conv2_1.weight, self.conv2_1.bias.unsqueeze(1), 2,3)))
        # x_cov4 = self.activation(self.bn2_2(self.CNN(x_cov3, self.conv2_2.weight, self.conv2_2.bias.unsqueeze(1), 4,5)))

         # third CNN
        x_cov3 = self.activation(self.bn3_1(self.CNN(x_cov2, self.conv3_1.weight, self.conv3_1.bias.unsqueeze(1), 3,4)))
        x_cov4 = self.activation(self.bn3_2(self.CNN(x_cov3, self.conv3_2.weight, self.conv3_2.bias.unsqueeze(1), 4,5)))
  
        # forth CNN
        x_cov5 = self.activation(self.bn4_1(self.CNN(x_cov4, self.conv4_1.weight, self.conv4_1.bias.unsqueeze(1), 5,6)))
        x_cov6 = self.activation(self.bn4_2(self.CNN(x_cov5, self.conv4_2.weight, self.conv4_2.bias.unsqueeze(1), 6,7)))
        
        # fc
        fc = x_cov6.view(-1, self.num_flat_features(x_cov6))
        
        fc1 = self.activation((self.linear(fc, self.fc1, 7,8)))
        
        fc2 = self.activation((self.linear(fc1, self.fc2, 8,9)))
        
        y = self.linear(fc2, self.fc3, 9,10)
        
        return y


    def w_norm(self, w, std_alpha = 10):
        # w_abs = torch.abs(w)
        std_w = torch.std(w)
        w_norm = np.abs(w/(std_alpha*std_w))
        # sum_w = torch.sum(w_abs)
        # min_vals = w_abs.min(dim=1, keepdim=True)[0]
        # max_vals = w_abs.max(dim=1, keepdim=True)[0]
        # w_minmax = ((w_abs - min_vals) / (max_vals - min_vals + 1e-6)) + 1e-6

        return w_norm
    
    
    def channel_norm(self, w, alpha=1.0):
        # w shape: [B, C, H, W] or [B, 1, H, W]

        w_abs = torch.abs(w)

        # reduce over spatial dims, not channel dim
        min_vals = w_abs.amin(dim=(2,3), keepdim=True)
        max_vals = w_abs.amax(dim=(2,3), keepdim=True)

        w_minmax = (w_abs - min_vals) / (max_vals - min_vals + 1e-6) + 1e-6

        return (1 - alpha) * w_abs + alpha * w_minmax
    
    
    # CNN using unfold/fold, calculate edges values
    def CNN_edges(self, ori, kernel, l1, l2, norm = 0):
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
            y = (ori_unf).transpose(1,2)
            edges = (y.unsqueeze(1) * w[None,i,:,:]).transpose(2,3)
            if norm == 1:
                edges = self.channel_norm(edges)

            res = edges if res == None else torch.cat((res, edges), axis=1)
            
        dim = res.shape[1]*res.shape[2]*res.shape[3]
        edges_v = res.reshape(-1, dim) # batch * edge_num

        return edges_v
    

    # fullyconnected
    def fc_edges(self, x_tmp, layer, l1, l2):
        mask = self.remove_mask[l1]
        mask = mask.to(self.device) 
            
        w = layer.weight.T

        cur_shape = layer.weight.shape[0]*layer.weight.shape[1]
        
        edge_v = torch.cat([torch.reshape(w * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)
        return edge_v
    
    
    def node_norm(self, x):
        x_flat = x.view(-1, self.num_flat_features(x))

        # Take absolute value and normalize per sample
        x_abs = torch.abs(x_flat)
        min_vals = x_abs.min(dim=1, keepdim=True)[0]
        max_vals = x_abs.max(dim=1, keepdim=True)[0]
        x_norm = (x_abs - min_vals) / (max_vals - min_vals)
        
        return x_flat
    

    # calculate edge weights
    def NN_info_batch(self, x):
        x = self.normalize(x)
        
        edge_value = None
        weights = None
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        nodes_before = x.view(-1, self.num_flat_features(x))
        nodes = self.node_norm(x)
        
        # CNN 1_1
        k1 = self.conv1_1.weight
        b1 = self.conv1_1.bias
        i,j = 1,2
        edge_v = (self.CNN_edges(x_tmp, k1, i,j)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k1, i,j)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.bn1_1(self.CNN(x, k1, b1.unsqueeze(1),i,j))
        nodes_before = torch.cat((nodes_before, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x = self.activation(x)

        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x_cov3.view(-1, self.num_flat_features(x_cov3))), axis = 1)

        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        # CNN 2_1
        k1 = self.conv2_1.weight
        b1 = self.conv2_1.bias
        i,j = 2,3
        edge_v = (self.CNN_edges(x_tmp, k1, i,j)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k1, i,j)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.bn2_1(self.CNN(x, k1, b1.unsqueeze(1),i,j))
        nodes_before = torch.cat((nodes_before, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x = self.activation(x)

        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x_cov3.view(-1, self.num_flat_features(x_cov3))), axis = 1)

        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)

        # CNN 3_1
        k1 = self.conv3_1.weight
        b1 = self.conv3_1.bias
        i,j = 3,4
        edge_v = (self.CNN_edges(x_tmp, k1, i,j)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k1, i,j)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.bn3_1(self.CNN(x, k1, b1.unsqueeze(1),i,j))
        nodes_before = torch.cat((nodes_before, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x = self.activation(x)
        
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x_cov3.view(-1, self.num_flat_features(x_cov3))), axis = 1)

        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        # CNN 3_2
        k1 = self.conv3_2.weight
        b1 = self.conv3_2.bias
        i,j = 4,5
        edge_v = (self.CNN_edges(x_tmp, k1, i,j)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k1, i,j)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.bn3_2(self.CNN(x, k1, b1.unsqueeze(1),i,j))
        nodes_before = torch.cat((nodes_before, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x = self.activation(x)
        
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x_cov3.view(-1, self.num_flat_features(x_cov3))), axis = 1)

        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        # CNN 4_1
        k1 = self.conv4_1.weight
        b1 = self.conv4_1.bias
        i,j = 5,6
        edge_v = (self.CNN_edges(x_tmp, k1, i,j)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k1, i,j)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.bn4_1(self.CNN(x, k1, b1.unsqueeze(1),i,j))
        nodes_before = torch.cat((nodes_before, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x = self.activation(x)

        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x_cov3.view(-1, self.num_flat_features(x_cov3))), axis = 1)

        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        # CNN 4_2
        k1 = self.conv4_2.weight
        b1 = self.conv4_2.bias
        i,j = 6,7
        edge_v = (self.CNN_edges(x_tmp, k1, i,j)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)
        
        ones = (self.CNN_edges(ones_tmp, k1, i,j)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.bn4_2(self.CNN(x, k1, b1.unsqueeze(1),i,j))
        nodes_before = torch.cat((nodes_before, x.view(-1, self.num_flat_features(x))), axis = 1)
        
        x = self.activation(x)
        
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x_cov3.view(-1, self.num_flat_features(x_cov3))), axis = 1)

        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)

        # fully connected
        x = x.view(-1, self.num_flat_features(x)) # batch * input size
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # fc1
        edge_v = (self.fc_edges(x_tmp, self.fc1, 7,8)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc1, 7,8)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc1(x)
        nodes_before = torch.cat((nodes_before, x), axis = 1)
        
        x = self.activation(x)
        
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x), axis = 1)
        
        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        # fc2    
        edge_v = (self.fc_edges(x_tmp, self.fc2, 8,9)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc2, 8,9)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc2(x)
        nodes_before = torch.cat((nodes_before, x), axis = 1)
        
        x = self.activation(x)
        
        x_tmp = x
        ones_tmp = torch.ones_like(x)
        
        # nodes = torch.cat((nodes, x), axis = 1)
        x_norm = self.node_norm(x)
        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)

        # fc3
        edge_v = (self.fc_edges(x_tmp, self.fc3, 9,10)).cpu().detach()
        edge_value = edge_v if edge_value == None else torch.cat((edge_value, edge_v), axis=1)

        ones = (self.fc_edges(ones_tmp, self.fc3, 9,10)).cpu().detach()
        # ones = self.w_norm(ones)
        weights = ones if weights == None else torch.cat((weights, ones), axis=1)
        
        x = self.fc3(x)
        nodes_before = torch.cat((nodes_before, x), axis = 1)
   
        # nodes = torch.cat((nodes, x), axis = 1)
        x_norm = self.node_norm(x)

        # Concatenate normalized x to nodes along feature dimension
        nodes = torch.cat((nodes, x_norm), axis=1)
        
        return edge_value, nodes, weights, nodes_before


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
    
    
    
    def normalization_weight_w4(self, nodes, weights, dims, model_dims, device='cuda'):
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
                p = cur_dim['padding'] if 'padding' in cur_dim else 0
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
                        
                        # node_slice = (node_slice - min_vals) / (max_vals - min_vals)
                        
                       # Prevent divide-by-zero
                        weights_inv2[:, in_edges] = 1./(torch.abs(node_slice))
                        
                        n += 1
                        start_col = end_col
                        end_col = start_col + step*pre_channel
                end_col = start_col
            
            elif cur_name == "fc":
                in_edges = torch.arange(start_col + (n - prefix_dims[current_l-1]), end_col, step)
            
                # Shape: (batch_size, fan-in)
                node_slice = torch.abs(nodes[:, neighbors])

                # Prevent divide-by-zero
                weights_inv2[:, in_edges] = 1./(torch.abs(node_slice))
                    
                n += 1

        return weights_inv1, weights_inv2

    
    
    # weights regularization 
    def normalization_weight_w4_old(self, nodes, weights, dims, model_dims):
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

                # print(len(out_neighbors), len(out_edges))

                node_slice = torch.abs(nodes[:, out_neighbors])
                # weights_inv3[:, out_edges] = 1.0 / torch.abs(out_node_slice)
                
                # --- Step 1: normalize to [0, 1] ---
                min_vals = node_slice.min(dim=1, keepdim=True)[0]
                max_vals = node_slice.max(dim=1, keepdim=True)[0]
                
                node_slice = (node_slice - min_vals) / (max_vals - min_vals)

                # --- Step 2: compute weights ---
                weights_inv3[:, out_edges] = 1.0 / (torch.abs(node_slice))

                n += 1

        
        return weights_inv1, weights_inv2, weights_inv3