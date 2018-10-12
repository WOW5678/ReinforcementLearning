# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/11 0011 下午 9:14
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

display_reward_threshold=-2000
render=False

env=gym.make('MountainCar-v0')
env.seed(1)
env=env.unwrapped

# 显示可用的action
print(env.action_space)
#显示可用state的observation
print(env.observation_space)
# 显示state的最高值
print(env.observation_space.high)
# 显示state的最低值
print(env.observation_space.low)

RL=PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.05,
    reward_decay=0.995
    #ouput_graph=True #输出tensorboard文件
)
# 在计算机跑完一整个回合后才更新一次
for i_episode in range(3000):
    observation=env.reset()

    while True:
        if render:
            env.render()
        #观察值就是神经网络的输入
        #observation是[-0.43852191  0.        ]
        #action=1
        action=RL.choose_action(observation)
        #observation_:[-0.43915308 -0.00063117]
        #reward:-1.0,done:False,info:{}
        observation_,reward,done,info=env.step(action)
        # 存储这一个回合的transition
        RL.store_transition(observation,action,reward)

        if done:
            ep_rs_sum=sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward=ep_rs_sum
            else:
                running_reward=running_reward*0.99+ep_rs_sum*0.01
            #判断是否显示模拟
            if running_reward>display_reward_threshold:
                render=True
            print('episode:',i_episode,'reward:',int(running_reward))

            # 学习，输出vt,
            vt=RL.learn()

            if i_episode==30:
                plt.plot(vt) # 画出这个回合的vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
                pass
            break
        observation=observation_
