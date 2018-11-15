# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
import sys, os
import json
import argparse
from myParser import Parser
#import parser
from datamanager import DataManager
from actor import ActorNetwork
from LSTM_critic import LSTM_CriticNetwork
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#get parse
argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
random.seed(args.seed)

#get data
#dataManager = DataManager(args.dataset)
dataManager=DataManager('../AGnews')
train_data, dev_data, test_data = dataManager.getdata(args.grained, args.maxlenth)
word_vector = dataManager.get_wordvector(args.word_vector)

if args.fasttest == 1:
    train_data = train_data[:1000]
    dev_data = dev_data[:200]
    test_data = test_data[:200]

print ("train_data ", len(train_data))
print ("dev_data", len(dev_data))
print ("test_data", len(test_data))

def sampling_RL(sess, actor, inputs, vec, lenth, epsilon=0., Random=True):
    #print epsilon
    current_lower_state = np.zeros((1, 2*args.dim), dtype=np.float32)
    actions = []
    states = []
    #sampling actions
    #length为每个句子的长度（不包含填充字符）
    for pos in range(lenth):
        #predicted是一个行向量，两个元素，分别为action为0的概率和为1的概率
        predicted = actor.predict_target(current_lower_state, [vec[0][pos]])
        
        states.append([current_lower_state, [vec[0][pos]]])
        # 随机型策略
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < predicted[0] else 1)
            else:
                action = (1 if random.random() < predicted[0] else 0)
        else:
            # 决策型策略
            action = np.argmax(predicted)

        actions.append(action)
        # 若当前action为1，表示不删除这个单词
        if action == 1:
            #out_d是lstm cell的输出，current_lower_state是lstm cell的状态是lstm cell的输出，self.target_lower_cell_state1是lstm cell的状态
            out_d, current_lower_state = critic.lower_LSTM_target(current_lower_state, [[inputs[pos]]])

    #当以及句子中每个单词都被决策以后
    # Rinput保存着每个被选择的单词
    Rinput = []
    for (i, a) in enumerate(actions):
        if a == 1:
            Rinput.append(inputs[i])
    # Rlenth：被选择的单词个数
    Rlenth = len(Rinput)
    #如果所有单词都没有被选择 则选择倒数第二个单词
    # ？？？为什么不是选择倒数第一个单词呢？？？
    if Rlenth == 0:
        actions[lenth-2] = 1
        Rinput.append(inputs[lenth-2])
        Rlenth = 1

    Rinput += [0] * (args.maxlenth - Rlenth)
    return actions, states, Rinput, Rlenth

def train(sess, actor, critic, train_data, batchsize, samplecnt=5, LSTM_trainable=True, RL_trainable=True):
    print ("training : total ", len(train_data), "nodes.")
    random.shuffle(train_data)
    for b in range(len(train_data) // batchsize):
        datas = train_data[b * batchsize: (b+1) * batchsize]
        totloss = 0.
        #在每个batch开始之前
        # 把target_network中的参数赋值给network_params
        critic.assign_active_network()
        actor.assign_active_network()

        # 针对这个batch中的每个句子
        for j in range(batchsize):
            #prepare
            data = datas[j]
            inputs, solution, lenth = data['words'], data['solution'], data['lenth']

            #train the predict network
            if RL_trainable:
                actionlist, statelist, losslist = [], [], []
                aveloss = 0.
                for i in range(samplecnt):
                    # actions为每个单词的action结果,states为每个时刻（单词）的状态
                    # Rinput是每个被选择的单词组成的序列（后面用0填充）
                    # Rlenth是被选择的单词个数
                    actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, args.epsilon, Random=True)
                    actionlist.append(actions)
                    statelist.append(states)
                    #out是分类的概率，loss：损失值
                    out, loss = critic.getloss([Rinput], [Rlenth], [solution])

                    #这个loss 就是critic network 返回大reward值，用于更新actor network的参数
                    loss += (float(Rlenth) / lenth) **2 *0.05

                    aveloss += loss
                    losslist.append(loss)
                
                aveloss /= samplecnt
                totloss += aveloss
                grad = None
                if LSTM_trainable:
                    out, loss, _ = critic.train([Rinput], [Rlenth], [solution])

                for i in range(samplecnt):
                    for pos in range(len(actionlist[i])):
                        rr = [0., 0.]
                        #rewards值
                        rr[actionlist[i][pos]] = (losslist[i] - aveloss) * args.alpha
                        # g有四个元素是每个参数的梯度值，但最后一个元素为None
                        # 因为我们设置了stop_gradients=action_gradients
                        g = actor.get_gradient(statelist[i][pos][0], statelist[i][pos][1], rr)

                        #进行参数的更新
                        if grad == None:
                            grad = g
                        else:
                            grad[0] += g[0]
                            grad[1] += g[1]
                            grad[2] += g[2]
                #将累计的梯度应用到network_params
                actor.train(grad)
            else:
                #out是句子的分类结果
                out, loss, _ = critic.train([inputs], [lenth], [solution])
                totloss += loss

        # 每个batch之后更新一次网络参数
        if RL_trainable:
            #更新target_network中的参数
            actor.update_target_network()
            if LSTM_trainable:
                critic.update_target_network()
        else:
            #将self.network_params的参数否赋值给self.target_network_params参数
            critic.assign_target_network()
        if (b + 1) % 500 == 0:
            acc_test = test(sess, actor, critic, test_data, noRL= not RL_trainable)
            acc_dev = test(sess, actor, critic, dev_data, noRL= not RL_trainable)
            print ("batch ",b , "total loss ", totloss, "----test: ", acc_test, "| dev: ", acc_dev)


def test(sess, actor, critic, test_data, noRL=False):
    acc = 0
    for i in range(len(test_data)):
        #prepare
        data = test_data[i]
        inputs, solution, lenth = data['words'], data['solution'], data['lenth']
        
        #predict
        if noRL:
            #out是分类的概率值
            out = critic.predict_target([inputs], [lenth])
        else:
            actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, Random=False)
            out = critic.predict_target([Rinput], [Rlenth])
        # 对分类概率值直接取最大的索引 即为预测的标签值
        if np.argmax(out) == np.argmax(solution):
            acc += 1
    return float(acc) / len(test_data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    #model
    critic = LSTM_CriticNetwork(sess, args.dim, args.optimizer, args.lr, args.tau, args.grained, args.maxlenth, args.dropout, word_vector) 
    actor = ActorNetwork(sess, args.dim, args.optimizer, args.lr, args.tau)
    #print variables
    # for item in tf.trainable_variables():
    #     print (item.name, item.get_shape())
    
    saver = tf.train.Saver()
    
    #LSTM pretrain
    if args.RLpretrain != '':
        pass


    #预训练
    #直接将句子进行分类（不删除任何特定的word）得到预训练的参数
    elif args.LSTMpretrain == '':
        sess.run(tf.global_variables_initializer())
        for i in range(0, 2):
            #预训练表示模型
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, RL_trainable=False)
            critic.assign_target_network()
            acc_test = test(sess, actor, critic, test_data, True)
            acc_dev = test(sess, actor, critic, dev_data, True)
            print ("LSTM_only ",i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_base", global_step=i)
        print ("LSTM pretrain OK")
    else:
        # 完成了预训练之后 直接加载模型中的参数
        print ("Load LSTM from ", args.LSTMpretrain)
        saver.restore(sess, args.LSTMpretrain)
    
    print ("epsilon", args.epsilon)

    # 预训练强化学习模型
    # 直接将随机化的参数用于单词的action选择
    if args.RLpretrain == '':
        for i in range(0, 5):
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, LSTM_trainable=False)
            acc_test = test(sess, actor, critic, test_data)
            acc_dev = test(sess, actor, critic, dev_data)
            print ("RL pretrain ", i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_RLpre", global_step=i)
        print ("RL pretrain OK")
    else:
        print ("Load RL from", args.RLpretrain)
        saver.restore(sess, args.RLpretrain)

    ###训练
    for e in range(args.epoch):

        # 两个模型同时级联的训练，此时RL_trainable=True, LSTM_trainable也为True
        train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt)
        acc_test = test(sess, actor, critic, test_data)
        acc_dev = test(sess, actor, critic, dev_data)
        print ("epoch ", e, "----test: ", acc_test, "| dev: ", acc_dev)
        saver.save(sess, "checkpoints/"+args.name, global_step=e)


