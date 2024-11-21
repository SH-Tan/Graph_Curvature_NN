import torch


class Edge_Remove():
    def __init__(self, net, dims, input_size_1d, G, curvature, file_path):
        self.net = net
        self.dims = dims
        self.input_size = input_size_1d
        self.G = G
        self.curvature = curvature
        self.layer_num = len(dims) - 1
        self.file_path = file_path
        
    def e_remove(self, edge_set, mark):
        self.net.eval()
        for (n1, n2) in edge_set:
            if (n1 > n2):
                tmp = n1
                n1 = n2
                n2 = tmp
            cur_l = 0
            
            while (cur_l < self.layer_num and (n1 > (self.dims[cur_l]-1))):
                n1 -= self.dims[cur_l]
                n2 -= self.dims[cur_l]
                cur_l += 1
                
            assert(n2 >= 0)
            while (cur_l < self.layer_num and (n2 > (self.dims[cur_l]-1))):
                n2 -= self.dims[cur_l]
                cur_l += 1
                    
            with torch.no_grad():
                if (cur_l == self.layer_num):
                    self.net.layer_list[cur_l-1].weight.T[n1,n2] = 0
                else: 
                    self.net.layer_list[cur_l-1].f4.f4.weight.T[n1,n2] = 0
                    
        torch.save(self.net.state_dict(), self.file_path + mark)
       