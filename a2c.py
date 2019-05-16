# -*- coding: utf-8 -*-
"""
 @Time    : 2019/5/16 0016 下午 4:13
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym

# Hyper Parameters
STATE_DIM = 4
ACTION_DIM = 2
STEP = 3000
SAMPLE_NUMS = 30

class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        #在数学上等价于log(softmax(x))
        out = F.log_softmax(self.fc3(out))
        return out

class ValueNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def roll_out(actor_network,task,sample_nums,value_network,init_state):
    #task.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    #sample_nums是每个episode的尝试次数，即完成sample_nums次的尝试即为一个episode
    for j in range(sample_nums):
        states.append(state)
        #根据状态 actor网络执行行为
        log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        #执行不同ation的概率
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_ = task.step(action)
        #fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = task.reset()
            break
    if not is_done:
        #计算这一个episode的最后的reward值
        final_r = value_network(Variable(torch.Tensor([final_state]))).cpu().data.numpy()

    return states,actions,rewards,final_r,state

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    # init a task generator for data fetching
    task = gym.make("CartPole-v0")
    #init_state:是一个包含了4个元素的数组
    init_state = task.reset()

    # init value network
    # 即 critic network 网络的输出值是针对特定状态的Q值？？待确认
    value_network = ValueNetwork(input_size = STATE_DIM,hidden_size = 40,output_size = 1)
    value_network_optim = torch.optim.Adam(value_network.parameters(),lr=0.01)

    # init actor network
    actor_network = ActorNetwork(STATE_DIM,40,ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.01)

    steps =[]
    task_episodes =[]
    test_results =[]

    for step in range(STEP):
        states,actions,rewards,final_r,current_state = roll_out(actor_network,task,SAMPLE_NUMS,value_network,init_state)
        init_state = current_state
        #每个step就执行一次模型训练
        actions_var = Variable(torch.Tensor(actions).view(-1,ACTION_DIM))
        states_var = Variable(torch.Tensor(states).view(-1,STATE_DIM))

        # train actor network
        actor_network_optim.zero_grad()
        log_softmax_actions = actor_network(states_var)
        vs = value_network(states_var).detach()
        #print('vs:',vs.shape)# [batchsize,1]
        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))
        #print('qs:',qs.shape) #[batchsize]
        advantages = qs - vs
        #print('advantages:',advantages.shape)#[batchsize,batchsize]
        #分解一下 为了看清中间的运算过程
        #print(torch.sum(log_softmax_actions*actions_var,1).shape) #torch.Size([batchsize])
        #print((torch.sum(log_softmax_actions*actions_var,1)*advantages).shape) #torch.Size([batchsize,batchsize])
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        actor_network_optim.step()

        # train value network
        value_network_optim.zero_grad()
        target_values = qs
        values = value_network(states_var)
        criterion = nn.MSELoss()
        # print('values:',values.shape)
        # print('target_values:',target_values.shape)
        value_network_loss = criterion(values,target_values.view(-1,1))
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(),0.5)
        value_network_optim.step()

        # Testing
        if (step + 1) % 50== 0:
                result = 0
                test_task = gym.make("CartPole-v0")
                #相当于创造10个样本
                for test_epi in range(10):
                    state = test_task.reset()
                    for test_step in range(200):#测试的时候 每个episode走200步
                        softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state]))))
                        #print(softmax_action.data)
                        #测试时 选择概率最大的action,而不是按照概率执行action
                        action = np.argmax(softmax_action.data.numpy()[0])
                        next_state,reward,done,_ = test_task.step(action)
                        result += reward
                        state = next_state
                        if done:
                            break
                print("step:",step+1,"test result:",result/10.0)
                steps.append(step+1)
                test_results.append(result/10)

if __name__ == '__main__':
    main()