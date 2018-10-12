# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/11 0011 下午 9:15
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient(object):
    # 初始化时，给出相关参数，并创建一个神经网络
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.95,output_graph=False):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate #学习率
        self.gamma=reward_decay #reward衰减率

        #每个回合的观察值，每个回合的action,每个回合的奖励值
        self.ep_obs,self.ep_as,self.ep_rs=[],[],[] # 存储回合信息的list
        self._build_net() # 建立policy 神经网络
        self.sess=tf.Session()
        # 是否输出tensorboard 文件
        if output_graph:
            tf.summary.FileWriter('logs/',self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    # 建立policy gradient 神经网络
    # 因为是强化学习，并没有y label，取而代之的是我们选择的action
    def _build_net(self):
        with tf.name_scope('inputs'):
            # 接受observation
            self.tf_obs=tf.placeholder(tf.float32,[None,self.n_features],name='observations')
            # 接受我们在这个回合中选过的actions
            self.tf_acts=tf.placeholder(tf.int32,[None,],name='actions_num')
            #接收每个state-action所对应的value(通过reward计算)
            self.tf_vt=tf.placeholder(tf.float32,[None,],name='actions_value')
        # fc1
        layer=tf.layers.dense(inputs=self.tf_obs,
                              units=10, # 输出个数
                              activation=tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
                              bias_initializer=tf.constant_initializer(0.1),
                              name='fc1')
        #fc2
        all_act=tf.layers.dense(inputs=layer,
                               units=self.n_actions, #输出个数
                               activation=None, #之后加上softmax
                               kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
                               bias_initializer=tf.constant_initializer(0.1),
                               name='fc2'
                            )
        ## use softmax to convert to probability
        # 选择每种行为的概率
        self.all_act_prob=tf.nn.softmax(all_act,name='act_prob')
        with tf.name_scope('loss'):
            # 最大化总体reward(log_p*R)就是在最小化-log_p*R,而tf的功能中只有最小化
            # 所选action的概率-log值
            neg_log_prob=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,labels=self.tf_acts)
            # （vt=本reward+衰减的未来reward）引导参数的梯度下降
            loss=tf.reduce_mean(neg_log_prob*self.tf_vt)
        with tf.name_scope('train'):
            self.train_op=tf.train.AdamOptimizer(self.lr).minimize(loss)

    # 选择行为
    def choose_action(self,observation):
        # 所有action的概率
        #prob_weights:[[0.34983623 0.30167422 0.34848955]]
        prob_weights=self.sess.run(self.all_act_prob,feed_dict={self.tf_obs:observation[np.newaxis,:]})
        #根据概率选择action
        #action:1
        action=np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action

    # 存储回合transition
    #a:1,r:-1.0,s:[-0.43852191  0.        ]
    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    # 学习更新参数
    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm=self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op,feed_dict={
            self.tf_obs:np.vstack(self.ep_obs),
            self.tf_acts:np.array(self.ep_as),
            self.tf_vt:discounted_ep_rs_norm
        })
        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]
        return discounted_ep_rs_norm

    # 衰减回合的reward
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs=np.zeros_like(self.ep_rs)
        running_add=0
        for t in reversed(range(0,len(self.ep_rs))):
            running_add=running_add*self.gamma+self.ep_rs[t]
            discounted_ep_rs[t]=running_add
        # normalize episode rewards
        discounted_ep_rs-=np.mean(discounted_ep_rs)
        discounted_ep_rs/=np.std(discounted_ep_rs)
        return discounted_ep_rs



