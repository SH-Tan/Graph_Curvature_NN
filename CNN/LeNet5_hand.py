import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        #self.lr = 0.01
        self.loss_func = nn.CrossEntropyLoss()

        # CNN initialization
        self.k1 = nn.Parameter(torch.zeros(6, 1, 5, 5))
        self.bc1 = nn.Parameter(torch.zeros(6,1))  
        self.k2 = nn.Parameter(torch.zeros(16, 6, 5, 5))
        self.bc2 = nn.Parameter(torch.zeros(16,1)) 
        nn.init.uniform_(self.k1, -np.sqrt(1 / (1 * 5)), np.sqrt(1 / (1 * 5)))  
        nn.init.uniform_(self.bc1, -np.sqrt(1 / (1 * 5)), np.sqrt(1 / (1 * 5))) 
        nn.init.uniform_(self.k2, -np.sqrt(1 / (6 * 5)), np.sqrt(1 / (6 * 5))) 
        nn.init.uniform_(self.bc2, -np.sqrt(1 / (6 * 5)), np.sqrt(1 / (6 * 5))) 

        # Linear initialization
        self.w1 = nn.Parameter(torch.zeros(16*5*5, 120))
        self.w2 = nn.Parameter(torch.zeros(120, 84))
        self.w3 = nn.Parameter(torch.zeros(84, 10))
        self.b1 = nn.Parameter(torch.zeros(120)) 
        self.b2 = nn.Parameter(torch.zeros(84)) 
        self.b3 = nn.Parameter(torch.zeros(10))  
        nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.w3, gain=nn.init.calculate_gain('sigmoid')) 
        nn.init.uniform_(self.b1, -np.sqrt(1 / 16*5*5), np.sqrt(1 / 16*5*5))  
        nn.init.uniform_(self.b2, -np.sqrt(1 / 120), np.sqrt(1 / 120))  
        nn.init.uniform_(self.b3, -np.sqrt(1 / 84), np.sqrt(1 / 84))  

        self.optimizer = torch.optim.Adam(self.parameters())


    # average Pooling
    def avepooling(self, input, k_size = 2, stride = 2):
        #print(input.shape)
        batch = input.shape[0]
        channel = input.shape[1]

        out_s0 = (input.shape[2] - k_size) // stride + 1
        out_s1 = (input.shape[3] - k_size) // stride + 1

        i_unf = F.unfold(input, (k_size, k_size), stride=2) 

        i_unf = i_unf.view(batch, channel, k_size*k_size, -1)
        i_unf = i_unf.view(batch*channel, k_size*k_size, -1)

        out = torch.mean(i_unf, dim = 1, keepdim=True)

        out = out.view(batch,channel, out_s0, out_s1)

        return out


    def CNN(self, ori, kernel, b):
        out_size = (ori.shape[2] - kernel.shape[2])//1 +1
        ori_unf = F.unfold(ori,(kernel.shape[2],kernel.shape[3]))
        out = ori_unf.transpose(1,2).matmul(kernel.view(kernel.size(0),-1).t()).transpose(1,2)
        out = F.fold(out, (out_size,out_size), (1,1))

        out = out + b[None,:,:,None]

        return out


    def linear(self, x0):
        x1 = x0.mm(self.w1)
        x1 = (x1.t() + self.b1[:,None]).t() 
        h1 = self.sigmoid(x1)
        x2 = h1.mm(self.w2)
        x2 = (x2.t() + self.b2[:,None]).t()
        h2 = self.sigmoid(x2)
        x3 = h2.mm(self.w3)
        y_hat = (x3.t() + self.b3[:,None]).t()
        return y_hat


    def forward(self, x, y):
        x_cov1 = self.CNN(x, self.k1, self.bc1)
        x_pool1 = self.avepooling(x_cov1)
        x_1 = self.sigmoid(x_pool1)
        x_cov2 = self.CNN(x_1, self.k2, self.bc2)
        x_pool2 = self.avepooling(x_cov2)
        x_2 = self.sigmoid(x_pool2)

        x_l = x_2.view(x_2.shape[0], -1)
        y_hat = self.linear(x_l)

        loss = self.loss_func(y_hat, y)

        return loss, y_hat

    def opt(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()