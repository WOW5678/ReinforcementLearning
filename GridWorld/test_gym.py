# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/30 0030 下午 3:04
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import gym
env=gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
