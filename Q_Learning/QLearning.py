# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/30 0030 下午 7:19
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import numpy as np
import random
import matplotlib.pyplot as plt
gamma = 0.7

reward = np.array([[0, -10, 0, -1, -1],

                   [0, 10, -1, 0, -1],

                   [-1, 0, 0, 10, -1],

                   [-1, 0, -10, 0, 10]])

# 即要学习的Q-Table
# 4种可能的状态5种可能的行动
q_matrix = np.zeros((4, 5))

# 状态转移矩阵，-1为不可转移的状态
transition_matrix = np.array([[-1, 2, -1, 1, 1],

                              [-1, 3, 0, -1, 2],

                              [0, -1, -1 , 3, 3],

                              [1, -1, 2, -1, 4]])


valid_actions = np.array([[1, 3, 4],

                          [1, 2, 4],

                          [0, 3, 4],

                          [0, 2, 4]])

for i in range(1000):

    start_state = 0

    current_state = start_state

    # 达到3状态的话 这个episode结束，完成了这次目标
    while current_state != 3:

        action = random.choice(valid_actions[current_state])

        next_state = transition_matrix[current_state][action]

        future_rewards = []

        for action_nxt in valid_actions[next_state]:

            future_rewards.append(q_matrix[next_state][action_nxt])

        q_state = reward[current_state][action] + gamma * max(future_rewards)

        q_matrix[current_state][action] = q_state

        current_state = next_state

print('Final Q-table:')

print(q_matrix)