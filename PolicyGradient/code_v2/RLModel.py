# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/22 0022 下午 4:06
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 强化学习实现对病人的多标签分类问题
"""
import tensorflow as tf
import numpy as np
import dataProcess_step3

class Environment(object):
    def __init__(self,sentence_len):
        self.sentence_len=sentence_len
    def reset(self):
        self.num_selected=0
        self.current_step=0
        self.label_selected=[]

    def step(self,action,y_true):
        if action==1:
            self.num_selected+=1
            self.label_selected.append(self.current_step)
            if self.current_step in y_true:
                reward=1
            else:
                reward=-1
        else:
            if self.current_step not in y_true:
                reward=0
            else:
                reward=-1
        self.current_step+=1
        return reward

    # def get_reward(self,y_true):
    #     count=0
    #     ids=list(np.nonzero(y_true)[0])
    #     #print('ids:',ids)
    #     for item in self.label_selected:
    #         #print('item:',item)
    #         if item in ids:
    #             count+=1
    #     return count*2-(self.num_selected+len(ids))

def get_action(prob_value):
    value=prob_value[0]
    tmp=np.random.uniform()
    if tmp<value:
        return 1
    else:
        return 0

def train():
    # 处理输入，是RL 模型的核心
    input_x = tf.placeholder(dtype=tf.int32, shape=[None, max_len])
    input_y = tf.placeholder(dtype=tf.int32,shape=[1])

    W_emb = tf.get_variable(name='word_embedding', initializer=tf.random_normal_initializer(),
                            shape=[feature_vocab_size+2, emb_size])
    W_emb_y=tf.get_variable(name='word_embedding_y',initializer=tf.random_uniform_initializer(),
                            shape=[len(label2id),rnn_size])

    input_x_emb = tf.nn.embedding_lookup(W_emb, input_x)
    #input_y_emb:(1,32)
    input_y_emb = tf.nn.embedding_lookup(W_emb_y, input_y)
    #print('input_y_emb:',input_y_emb)

    input_x_emb = tf.transpose(input_x_emb, [1, 0, 2])
    # 将input_x_emb变成一个列表，列表中每个元素为（batch_size,emb_size）
    # 这样是为了满足lstm模型的输入格式
    input_x_lstm = tf.unstack(input_x_emb, axis=0)
    # 使用一个LSTM模型编码输入
    lstm_ceil = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=0.0, state_is_tuple=True)
    _state = lstm_ceil.zero_state(batch_size, tf.float32)
    outputs, final_state = tf.nn.static_rnn(lstm_ceil, input_x_lstm, initial_state=_state)

    # 句子表示为最后一个隐含特征
    sentence = outputs[-1]
    #句子表示与标签表示做点乘 作为匹配得分值
    prob = tf.contrib.layers.fully_connected(tf.multiply(sentence,input_y_emb), 1, tf.nn.sigmoid)

    # [-1],表示变成一个行向量 [batch_size]
    prob = tf.reshape(prob, [-1])
    # action=1 if np.random.random()<prob else 0
    # action=np.where(np.random.random()<prob,1,0)[0]
    reward_holder = tf.placeholder(dtype=tf.float32, shape=[None])
    action_holder=tf.placeholder(dtype=tf.float32,shape=[None])
    #pi = tf.stack([1.0 - prob, prob])
    ## the probability of choosing 0 or 1
    loglik = tf.log(action_holder * (action_holder - prob) + (1 - action_holder) * (action_holder + prob))
    # 计算损失值
    loss = -tf.reduce_sum(loglik * reward_holder)
    # 优化过程
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    # 更新参数

    #所有的参数
    tvars = tf.trainable_variables()


    # 计算梯度
    grads = tf.gradients(loss, tvars)
    print('grads:', grads)
    # update parameters using gradient
    gradient_holders = []
    for ix, var in enumerate(tvars):
        placeholder = tf.placeholder(tf.float32, name=str(ix) + '_holder')
        gradient_holders.append(placeholder)
        # grads_and_vars: List of (gradient, variable) pairs
    update_batch = optimizer.apply_gradients(zip(gradient_holders, tvars))


    with tf.Session() as sess:
        env=Environment(max_len)
        sess.run(tf.global_variables_initializer())
        gradBuffer=sess.run(tvars)
        for i in range(epoch_numbers):
            for ix, grad in enumerate(gradBuffer):
                # 将所有的值都变为0
                gradBuffer[ix] = grad * 0
            #print('gradbuffer:',gradBuffer)
            # 针对每个句子，与每个标签进行
            for idx in range(len(train_data)):
                reward_sum=0
                env.reset()
                predict_list = []
                for j in range(len(label2id)):
                    prob_value = sess.run(prob, feed_dict={input_x: [train_data[idx]],input_y:[j]})
                    #('prob_value:',prob_value)
                    action = 1 if np.random.random()<prob_value else 0
                    y = 1 if action == 0 else 0
                    predict_list.append(y)
                    reward_sum+=env.step(action,train_label[idx])
                print('epoch:%d,reward_sum:%d'%(i,reward_sum))

                # 梯度的累计
                # 计算梯度值
                for j in range(len(label2id)):
                    tGrads=sess.run(grads,feed_dict={input_x:[train_data[idx]],input_y:[j],reward_holder:[reward_sum],action_holder:[predict_list[j]]})
                    #print('tGrads:',tGrads)
                    for ix, grad in enumerate(tGrads):
                        if ix==0 or ix==1:
                            indice = grad.indices[0]
                            gradBuffer[ix][indice] += grad.values[0]
                        else:
                            gradBuffer[ix] =gradBuffer[ix]+ grad
                #所有标签的梯度都进行了累计 此时进行参数的更新
                sess.run(update_batch,feed_dict={gradient_holders[0]:gradBuffer[0],gradient_holders[1]:gradBuffer[1],
                                                 gradient_holders[2]:gradBuffer[2],gradient_holders[3]:gradBuffer[3],
                                                 gradient_holders[4]:gradBuffer[4],gradient_holders[5]:gradBuffer[5]})
            #
            # 每50轮之后 计算一下预测的标签
            if i%10==0 or i==epoch_numbers:
                recall_list=[]
                for idx in range(len(test_data)):

                    action_list_test=[]
                    predict_list_test=[]
                    env.reset()
                    reward_sum_test=0
                    for idy in range(len(label2id)):
                        prob_value=sess.run(prob,feed_dict={input_x:[features[idx]],input_y:[idy]})
                        action=get_action(prob_value)
                        y=1 if action==0 else 0
                        action_list_test.append(action)
                        predict_list_test.append(y)
                        reward_sum_test+=env.step(action,test_labels[idx])

                    print('epoch:%d,reward_sum_test:%d'%(i,reward_sum_test))
                    # print('true_label:',true_label)
                    # print('action_label:',action_list_test)
                    #print('predict_label:',predict_list_test)

                    # 统计Recall值
                    recall=get_recall(test_labels[idx], action_list_test)
                    recall_list.append(recall)
                average_recall=sum(recall_list)/len(recall_list)
                print('average_recall is:',average_recall)

def get_recall(true_label,action_list_test):
    action_list_test=np.array(action_list_test)
    predicted = list(np.nonzero(action_list_test)[0])
    print('true_label:',true_label)
    print('predicted:',predicted)
    correct=[item  for item in predicted if item in true_label]

    recall=1.0*len(correct)/len(true_label)
    print('recall:',recall)
    return recall

def split_train_test(data,labels,train_rate):
    ids=np.random.permutation(len(data))
    data=np.array(data)
    labels=np.array(labels)
    data=data[ids]
    labels=labels[ids]
    train_num=int(len(data)*train_rate)
    train_data=data[:train_num]
    train_label=labels[:train_num]
    test_data=data[train_num:]
    test_labels=labels[train_num:]
    return train_data,train_label,test_data,test_labels


if __name__ == '__main__':
    # STEP 1: 定义模型参数
    emb_size=50
    max_len=70
    rnn_size=32
    batch_size=1
    lr=0.001
    epoch_numbers=1000

    # STEP 2: 预处理数据，将病人特征id化，将标签one-hot编码

    # patientFeatures:['头晕 疼痛 右下肢'，'....’]
    # patientLabels:['鼻骨骨折 面部外伤 高血压','....']
    patientFeatures, patientLabels = dataProcess_step3.load_data(filepath='../filtered_data_2.csv')
    feature2id, id2feature, feature_vocab_size = dataProcess_step3.build_vocab(patientFeatures, max_vocab_size=20000)
    label2id, id2label = dataProcess_step3.build_vocab_label(patientLabels)
    features = dataProcess_step3.transform2ids(patientFeatures,feature2id,max_len)
    #print('features:', features)
    #labels = dataProcess.labelEncoder(patientLabels,label2id)
    labels=dataProcess_step3.labelEncoder_2(patientLabels,label2id)
    print('labels:',labels)
    # STEP 3: 训练RL模型（包含RL模型的过程）
    # for idx in range(len(features)):
    #     #true_label = np.nonzero(labels[idx])
    #     true_label=labels[idx]
    #     print('true_label:',true_label)
    train_data, train_label, test_data, test_labels=split_train_test(features[:2000],labels[:2000],0.3)
    train()




