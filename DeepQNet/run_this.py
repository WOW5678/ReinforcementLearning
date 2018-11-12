# -*- coding:utf-8 -*-
'''
Create time: 2018/11/11 16:51
@Author: 大丫头

程序的入口
'''
from maze_env import Maze
from RL_brain import DeepQNetwork


# DQN与环境交互最重要的部分
def run_maze():
    step = 0  # 用来控制什么时候学习
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # DQN存储记忆
            RL.store_transition(observation, action, reward, observation_)
            #控制学习起始的时间和频率（先积累一些经验再开始学习）
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            #将下一个state_变为下次循环的state
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1  #总步数

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200, #每200步替换一次target_net的参数
                      memory_size=2000, #记忆上线
                      # output_graph=True #是否输出tensorboard文件
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()