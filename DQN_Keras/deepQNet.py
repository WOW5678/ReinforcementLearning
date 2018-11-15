# -*- coding:utf-8 -*-
'''
Create time: 2018/11/15 22:14
@Author: 大丫头
使用keras框架实现一个简单的强化学习实例

'''
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import  Adam

EPISODES=1000# 让agent玩游戏的次数

class DQNAgent(object):
    def __init__(self,state_size,actioon_size):
        self.state_size=state_size
        self.action_size=actioon_size  # 两个值，为0或者1，分别表示向左向右
        self.memory=deque(maxlen=2000)
        self.gamma=0.95
        self.epsilon=1.0 # agent 最初探索环境时选择 action 的探索率
        self.epsilon_min=0.01 # agent 控制随机探索的阈值
        self.epsilon_decay=0.995 # 随着 agent 玩游戏越来越好，降低探索率
        self.learning_rate=0.001
        self.model=self._build_model()

    def _build_model(self):
        model=Sequential()
        model.add(Dense(24,input_dim=self.state_size,activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self,batch_size):
        minibatch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                #target 为目标值
                target=(reward+self.gamma*(np.amax(self.model.predict(next_state)[0])))

            #target_f 为前面建立的神经网络的输出，是预测值，也就是损失函数里的 Q(s,a)
            target_f=self.model.predict(state)
            #print('targe_f:',target_f)
            # ？？？为什么进行更新呢？？？我不明白损失函数是怎么计算的？？？
            target_f[0][action]=target
            #然后模型通过 fit() 方法学习输入输出数据对
            self.model.fit(state,target_f,epochs=1,verbose=0)

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

if __name__ == '__main__':
    # 初始化gym环境和agent
    env=gym.make('CartPole-v1')
    state_size=env.observation_space.shape[0]
    action_size= env.action_space.n
    agent=DQNAgent(state_size,action_size)

    done=False
    batch_size=32

    # 开始迭代游戏
    for e in range(EPISODES):
        #每次游戏开始都需要重新设置一下状态
        state=env.reset()
        state=np.reshape(state,[1,state_size])

        #time表示游戏的每一帧
        #每成功保持杆子平衡一次得分就加1，最高到500分
        #目标是希望得分越高越好

        for time in range(500):
            #每一帧时，agent 根据state选择action
            action=agent.act(state)
            #这个action使得游戏进入下一个状态next_state，并且拿到了奖励reward
            #如果杆依旧平衡，reward为1，游戏结束则为-10
            next_state,reward,done,_=env.step(action)
            reward=reward if not done else -10
            next_state=np.reshape(next_state,[1,state_size])

            # 记忆之前的信息：state,action,reward,and done
            agent.remember(state,action,reward,next_state,done)

            #更新下一帧的所在状态
            state=next_state
            #如果杆倒了 则游戏结束，打印分数
            if done:
                # 可以通过观察time的值发现 time值越来越大，说明这个cartplor越来越会保持平衡了
                print('episode:{}/{},score:{},e:{:.2f}'.format(e,EPISODES,time,agent.epsilon))
                break
            #用之前的经验训练agent
            if len(agent.memory)>batch_size:
                agent.replay(batch_size)



