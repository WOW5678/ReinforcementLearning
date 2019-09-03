# -*- coding:utf-8 -*-
"""
@Time: 2019/09/03 10:56
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 实现分层的DQN网络
"""
import math
import random
from  collections import namedtuple,deque

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from IPython.display import clear_output
import matplotlib.pyplot as plt

USE_CUDA=torch.cuda.is_available()
Variable=lambda *args,**kwargs:autograd.Variable(*args,**kwargs).cuda() if USE_CUDA else autograd.Variable(*args,**kwargs)


class StochasticMDP:
    def __init__(self):
        self.end=False
        self.current_state=2
        self.num_actions=2
        self.num_state=6
        self.p_right=0.5

    ## 重置环境的状态
    def reset(self):
        self.end=False
        self.current_state=2
        state=np.zeros(self.num_state)
        state[self.current_state-1]=1.
        return state

    ## 将action作用到环境上，得到reward 并且改变状态
    def step(self,action):
        if self.current_state!=1:
            if action==1:
                if random.random()<self.p_right and self.current_state<self.num_state:
                    self.current_state+=1
                else:
                    self.current_state-=1

            if action==0:
                self.current_state -=1

            if self.current_state==self.num_state:
                self.end=True

        state=np.zeros(self.num_state)
        state[self.current_state-1]=1.0

        if self.current_state==1:
            if self.end:
                return state,1.00,True,{}
            else:
                return state,1.00/100.00,True,{}
        else:
            return state,0.0,False,{}

class ReplayBuffer(object):
    def __init__(self,capacity):
        self.capacity=capacity
        self.buffer=deque(maxlen=capacity)

    ## 将agent的执行过程保存到经验池中
    def push(self,state,action,reward,next_state,done):
        state=np.expand_dims(state,0)
        next_state=np.expand_dims(next_state,0)
        self.buffer.append((state,action,reward,next_state,done))

    ## 从经验池中随机取出 batch_size个样本
    def sample(self,batch_size):
        state,action,reward,next_state,done=zip(*random.sample(self.buffer,batch_size))
        return np.concatenate(state),action,reward,np.concatenate(next_state),done

    def __len__(self):
        return len(self.buffer)


class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Net,self).__init__()

        self.layers=nn.Sequential(
            nn.Linear(num_inputs,256),
            nn.ReLU(),
            nn.Linear(256,num_outputs)
        )

    def forward(self,x):
        return self.layers(x)

    ## 根据环境状态， agent选出要做出的action
    def act(self,state,epsilon):
        if random.random()>epsilon:
            state=torch.FloatTensor(state).unsqueeze(0)
            action=self.forward(Variable(state,volatile=True)).max(1)[1]
            return action.data[0]
        else:
            return random.randrange(num_actions)

def to_onehot(x):
    oh=np.zeros(6)
    oh[x-1]=1.0
    return oh


## 更新网络参数
def update(model,optimizer,replay_buffer,batch_size):
    if batch_size>len(replay_buffer):
        return
    state,action,reward,next_state,done=replay_buffer.sample(batch_size)

    state=Variable(torch.FloatTensor(state))
    next_state=Variable(torch.FloatTensor(next_state),volatile=True)
    action=Variable(torch.LongTensor(action))
    reward=Variable(torch.FloatTensor(reward))
    done=Variable(torch.FloatTensor(done))

    q_value=model(state)
    q_value=q_value.gather(1,action.unsqueeze(1)).squeeze(1)

    next_q_value=model(next_state).max(1)[0]
    expected_q_value=reward+0.99*next_q_value*(1-done)

    loss=(q_value-Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    env = StochasticMDP()
    num_goals = env.num_state  # 子目标的个数
    num_actions = env.num_actions

    # 低级别的net,输出向量长度应该为action的个数
    model = Net(2 * num_goals, num_actions)
    target_model = Net(2 * num_goals, num_actions)

    # 高级别的net 输出应该为goal的个数 而不是action的个数
    meta_model = Net(num_goals, num_goals)
    target_meta_model = Net(num_goals, num_goals)

    if USE_CUDA:
        model = model.cuda()
        target_model = target_model.cuda()
        meta_model = meta_model.cuda()
        target_meta_model = target_meta_model.cuda()

    optimizer = optim.Adam(model.parameters())
    meta_optimizer = optim.Adam(meta_model.parameters())

    replay_buffer = ReplayBuffer(10000)
    meta_replay_buffer = ReplayBuffer(10000)

    epsilon_start=1.0
    epsilon_final=0.01
    epsilon_decay=500 #衰减频率
    epsilon_by_frame=lambda frame_idx:epsilon_final+(epsilon_start-epsilon_final)*math.exp(-1.0*frame_idx/epsilon_decay)

    num_frames=100000
    frame_idx=1

    state=env.reset()
    done=False
    all_rewards=[]
    episode_reward=0

    while frame_idx<num_frames:
        ## 高级别的agent 根据状态 选择出子目标
        goal=meta_model.act(state,epsilon_by_frame(frame_idx))
        onehot_goal=to_onehot(goal)

        #将当前的环境状态作为高级别agent的环境状态
        meta_state=state
        extrinsic_reward=0 #外来的reward，作为衡量高级别agent的指标

        ## goal!=np.argmax(state) 意味着更改了子目标
        while not done and goal!=np.argmax(state):
            # 低级别agent 的状态包括两部分 作为环境状态的表示
            goal_state=np.concatenate([state,onehot_goal])

            # 低级别agent 根据环境状态选择action
            action=model.act(goal_state,epsilon_by_frame(frame_idx))
            # 环境针对该action 给出奖惩 并改变环境状态（此处的环境状态不包含目标表示，因此需要拼接成目标表示才能得到作为低目标agent选择action的表示）
            next_state, reward, done, _=env.step(action)

            episode_reward+=reward

            # 低级别agent得到的奖励 都会作为高级别agent的奖励
            extrinsic_reward+=reward

            # 这是低级别agent的奖励，用来调控低级别agent的参数
            intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0

            replay_buffer.push(goal_state,action,intrinsic_reward,np.concatenate([next_state,onehot_goal]),done)

            state=next_state

            # 每次更新都要同时更新高级别和低级别的agent参数
            update(model,optimizer,replay_buffer,32)
            update(meta_model,meta_optimizer,meta_replay_buffer,32)
            frame_idx+=1

            # 展示reward的变化情况
            if frame_idx%10000==0:
                clear_output(True)
                n=100
                plt.figure(figsize=(10,5))
                plt.title(frame_idx)
                plt.plot([np.mean(all_rewards[i:i+n]) for i in range(0,len(all_rewards),n)])
                plt.show()

        # 将高级别agent的执行轨迹加入到经验池中
        meta_replay_buffer.push(meta_state,goal,extrinsic_reward,state,done)

        #每个episode结束时 重置环境 并统计对应的reward
        if done:
            state=env.reset()
            done=False
            all_rewards.append(episode_reward)
            episode_reward=0






