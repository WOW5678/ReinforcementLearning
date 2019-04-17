"""
@author: orrivlin
"""

import torch 
import torch.nn
import torch.nn.functional as F


class NN(torch.nn.Module):
    def __init__(self,in_dim,out_dim,n_hid):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid
        
        self.fc1 = torch.nn.Linear(in_dim,n_hid,'linear')
        self.fc2 = torch.nn.Linear(n_hid,n_hid,'linear')
        self.fc3 = torch.nn.Linear(n_hid,out_dim,'linear')
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        #y = self.softmax(y)
        return y


class RND:
    def __init__(self,in_dim,out_dim,n_hid):
        #target网络是随机初始的网络 并且一直没有被更新 只用来对比与model网络的差异
        #若差异越大 说明agent探索的是一个新的state,应该得到更多的奖励
        #否则，说明agent当前执行的state与之前的比较相似，探索能力变弱，应该得到较小的reward
        self.target = NN(in_dim,out_dim,n_hid)
        self.model = NN(in_dim,out_dim,n_hid)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        
    def get_reward(self,x):
        y_true = self.target(x).detach()
        y_pred = self.model(x)
        reward = torch.pow(y_pred - y_true,2).sum()
        return reward

    #更新的是随机神经蒸馏网络的参数
    #target的参数仍然不发生改变
    def update(self,Ri):
        Ri.sum().backward()
        self.optimizer.step()
        
