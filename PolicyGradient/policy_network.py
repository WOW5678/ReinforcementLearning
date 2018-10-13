# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/12 0012 下午 8:07
 @Author  : Shanshan Wang
 @Version : Python3.5
"""

from __future__ import division
import numpy as np
import tensorflow as tf

# Load the CarPole Environment
import gym
env=gym.make('CartPole-v0')

#What happens if we try running the environment with random actions? How well do we  do? (Hint: not so well.)
# env.reset()
# random_episodes=0
# # reward_sum 保存一个episode的reward值
# reward_sum=0
# while random_episodes<10:
#     env.render()
#     # step函数传入的是action
#     observation,reward,done,_=env.step(np.random.randint(0,1))
#     reward_sum+=reward
#     if done:
#         random_episodes+=1
#         print('Reward for this episode was:',reward_sum)
#         reward_sum=0
#         env.reset()

# setting up neural network agent
# hyperparameters
H=50 # number of hidden layer neurons
# every how many episode to do a para update
batch_size=25
learning_rate=1e-2 # feel free to play with this to train faster or more stably
gamma=0.99 # discount factor for reward

# input dimensionality
D=4

tf.reset_default_graph()

##This defines the network as it goes from taking an observation of the environment to
#giving a probability of chosing to the action of moving left or right.
observations=tf.placeholder(tf.float32,[None,D],name='input_x')
w1=tf.get_variable('w1',shape=[D,H],
                   initializer=tf.contrib.layers.xavier_initializer())
layer1=tf.nn.relu(tf.matmul(observations,w1))
w2=tf.get_variable('w2',shape=[H,1],
                   initializer=tf.contrib.layers.xavier_initializer())
score=tf.matmul(layer1,w2)
probability=tf.nn.sigmoid(score)

# From here we define the parts of the network needed for learning a good policy
# tvars是一个列表，第一个元素是w1的参数，第二个元素是w2的参数
tvars=tf.trainable_variables()
input_y=tf.placeholder(tf.float32,[None,1],name='input_y')
advantages=tf.placeholder(tf.float32,name='reward_signal')

# Loss function
# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely
# 没看懂这个损失值？？？？？
#而且通过这个训练，rewards的值是下降的，肯定是损失值或者参数更新方向有问题
#loglik，Action取值 1概率probability(策略网络输出概率)，Action取值 0概率 1-probability。label取值，
# label=1-Action。Action 1，label 0，loglik=tf.log(probability)，Action取值为1的概率对数。Action 0，label 1，
# loglik=tf.log(1-probability)，Action取值为0的概率对数。loglik，当前Action对应概率对数。loglik与潜在坐advantages相乘，
# 取负数作损失，优化目标
loglik=tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))
loss = -tf.reduce_mean(loglik*advantages)
#newGrads是一个列表，第一个元素是w1的梯度值，第二个元素是w2的梯度值
newGrads=tf.gradients(loss,tvars)

## Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam=tf.train.AdamOptimizer(learning_rate=learning_rate)
## Placeholders to send the final gradients through when we update
w1grad=tf.placeholder(tf.float32,name='batch_grad1')
w2grad=tf.placeholder(tf.float32,name='batch_grad2')
batch_grad=[w1grad,w2grad]
updateGrads=adam.apply_gradients(zip(batch_grad,tvars))


def dicount_rewards(r):
    '''
    take ID float array rewards and compute discounted reward
    :param r:
    :return:
    '''
    discounted_r=np.zeros_like(r)
    running_add=0
    for t in reversed(range(0,r.size)):
        running_add=running_add*gamma+r[t]
        discounted_r[t]=running_add
    return discounted_r

# Running the agent and environment
xs,hs,dlogps,drs,ys,tfps=[],[],[],[],[],[]
running_reward=None
reward_sum=0
episode_number=1
total_episodes=10000
init=tf.global_variables_initializer()

# Launch the graph
with tf.Session()as sess:
    rendering=False
    sess.run(init)
    # 获取环境的一个初始的观测值
    #观测值是一个[1,4]的数组
    observation=env.reset()

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    #gradeBuffer是一个列表，包含该两个元素，第一个元素是w1，第二个元素是w2
    gradBuffer=sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        #将所有的值都变为0
        gradBuffer[ix]=grad*0

    while episode_number<=total_episodes:
        if reward_sum/batch_size>100 or rendering==True:
            env.render()
            rendering=True

        # 确保观察值的形状是网络可以处理的形状
        x=np.reshape(observation,[1,D])

        tfprob=sess.run(probability,feed_dict={observations:x})
        action=1 if np.random.uniform()<tfprob else 0

        xs.append(x) # observation
        y=1 if action==0 else 0# a fake label
        ys.append(y)

        # step the environment and get new measurement
        observation,reward,done,info=env.step(action)
        reward_sum+=reward

        #record reward (has to be done after we call step() to get reward for previous action)
        drs.append(reward)

        if done:
            episode_number+=1
            epx=np.vstack(xs)
            epy=np.vstack(ys)
            epr=np.vstack(drs)
            tfp=tfps
            xs,hs,dlogps,drs,ys,tfps=[],[],[],[],[],[]

            # compute the discounted reward backwards through time
            discounted_epr=dicount_rewards(epr)
            # size the rewards to unit normal (helps control  the gradient estimator variance)
            discounted_epr-=np.mean(discounted_epr)
            discounted_epr//=np.std(discounted_epr)

            # get the gradient for the episode and save it in gradbuffer
            tGrad=sess.run(newGrads,feed_dict={observations:epx,input_y:epy,advantages:discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix]+=grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number%batch_size==0:
                sess.run(updateGrads,feed_dict={w1grad:gradBuffer[0],w2grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix]=grad*0

                ## Give a summary of how well our network is doing for each batch of episodes.
                running_reward=reward_sum if running_reward is None else running_reward*0.99 +reward_sum*0.01
                print('Average reward for episode %f. Total average reward %f.'%(reward_sum//batch_size,running_reward//batch_size))
                if reward_sum//batch_size>200:
                    print('Task solved in',episode_number,'episode')
                    break
                reward_sum=0

            observation=env.reset()

    print(episode_number,' Episodes completed/')

