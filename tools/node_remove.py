import numpy as np
import torch

class Node_Remove():
    def __init__(self, net, dims, rem_n, file_path):
        self.net = net
        self.dims = dims
        self.layer_num = len(dims) - 1
        self.file_path = file_path
        self.num = rem_n
        self.predim = np.cumsum([0] + dims).tolist()

    def n_remove(self, node_set, mark):
        self.net.eval()
        removed = 0

        for node in node_set:
            if removed >= (int)(self.num):
                break

            n = node

            layer = np.searchsorted(self.predim, n, side='right') - 1
            local_idx = n - self.predim[layer]
            # Skip input and output layer nodes (optional: allow output)
            if layer == 0 or layer == self.layer_num:
                continue

            with torch.no_grad():
                if layer == self.layer_num - 1:
                    self.net.layer_list[layer - 1].f4.f4.weight[local_idx, :] = 0.0

                    self.net.layer_list[layer].weight[:, local_idx] = 0.0
                else:
                    self.net.layer_list[layer - 1].f4.f4.weight[local_idx, :] = 0.0

                    self.net.layer_list[layer].f4.f4.weight[:, local_idx] = 0.0

            removed += 1


        torch.save(self.net.state_dict(), self.file_path + mark)
