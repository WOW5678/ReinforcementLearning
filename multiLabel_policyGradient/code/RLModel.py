# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/22 0022 下午 4:06
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 强化学习实现对病人的多标签分类问题
"""
import tensorflow as tf
import numpy as np
import dataProcess



class Environment(object):
    def __init__(self,sentence_len):
        self.sentence_len=sentence_len
    def reset(self):
        self.num_selected=0
        self.current_step=0
        self.label_selected=[]

    def step(self,action,y_true,):
        if action==1:
            self.num_selected+=1
            self.label_selected.append(self.current_step)
        self.current_step += 1
        new_state=self.label_selected
        if self.current_step<len(label2id):
            reward=0
        else:
            reward=self.get_reward(y_true)

        return new_state,reward

    def get_reward(self,y_true):
        count=0
        ids=list(np.nonzero(y_true)[0])
        #print('ids:',ids)
        for item in self.label_selected:
            #print('item:',item)
            if item in ids:
                count+=1
        return count*2-(self.num_selected+len(ids))

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
    input_y = tf.placeholder(dtype=tf.float32, shape=[None, len(label2id)])

    W_emb = tf.get_variable(name='word_embedding', initializer=tf.random_normal_initializer(),
                            shape=[feature_vocab_size+2, emb_size])
    input_x_emb = tf.nn.embedding_lookup(W_emb, input_x)
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
    prob = tf.contrib.layers.fully_connected(sentence, 1, tf.nn.sigmoid)
    # [-1],表示变成一个行向量 [batch_size]
    prob = tf.reshape(prob, [-1])

    reward_holder = tf.placeholder(dtype=tf.float32, shape=[None])
    action_holder = tf.placeholder(dtype=tf.float32, shape=[None])

    ## the probability of choosing 0 or 1
    pi = action_holder * prob + (1 - action_holder) * (1 - prob)

    # 计算损失值
    loss = -tf.reduce_sum(tf.log(pi) * reward_holder)
    # 优化过程
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    # 更新参数
    tvars = tf.trainable_variables()

    # self.tvars_holders 是需要feed的placeholder
    tvars_holders = []
    for ix, var in enumerate(tvars):
        placeholder = tf.placeholder(dtype=tf.float32, name=str(ix) + '_holder')
        tvars_holders.append(placeholder)
    update_vars = []
    for ix, var in enumerate(tvars):
        update_var = tf.assign(var, tvars_holders)
        update_vars.append(update_var)

    # 计算梯度
    grads = tf.gradients(loss, tvars)
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

            # 针对每个句子，与每个标签进行
            for sent_i in features:
                action_list=[]
                reward_list=[]
                env.reset()
                prob_value = sess.run(prob, feed_dict={input_x: [sent_i]})
                for j in range(len(label2id)):
                    action = get_action(prob_value)
                    action_list.append(action)
                    new_state,reward=env.step(action,labels[j])
                #print('reward:',reward)
                reward_list.append(reward)

                # 所有的标签都整过一遍之后，进行参数的更新
                # 计算梯度值
                tGrads=sess.run(grads,feed_dict={input_x:[sent_i],input_y:[labels[j]],reward_holder:reward_list,action_holder:action_list})

                for ix, grad in enumerate(tGrads):
                    if ix==0:
                        indice = grad.indices[0]
                        gradBuffer[0][indice] += grad.values[0]
                    else:
                        gradBuffer[ix] =gradBuffer[ix]+ grad
            sess.run(update_batch,feed_dict={gradient_holders[0]:gradBuffer[0],gradient_holders[1]:gradBuffer[1],
                                             gradient_holders[2]:gradBuffer[2],gradient_holders[3]:gradBuffer[3],
                                             gradient_holders[4]:gradBuffer[4]})

            # 每50轮之后 计算一下预测的标签
            if i%50==0 or i==epoch_numbers:
                for  idx in range(len(features)):
                    env.reset()
                    true_label=np.nonzero(labels[idx])[0]
                    for j in range(len(label2id)):
                        prob_value=sess.run(prob,feed_dict={input_x:[features[idx]]})
                        action=get_action(prob_value)
                        action_list.append(action)
                        new_state,reward=env.step(action,labels[j])
                    prediction=new_state
                    print('epoch:',i)
                    print('reward:',reward)
                    print('true_label:',true_label)
                    print('prediction:',prediction)
                    # print('len(prediction:)',len(prediction))


if __name__ == '__main__':
    # STEP 1: 定义模型参数
    emb_size=50
    max_len=30
    rnn_size=32
    batch_size=1
    lr=0.001
    epoch_numbers=1000

    # STEP 2: 预处理数据，将病人特征id化，将标签one-hot编码

    # patientFeatures:['头晕 疼痛 右下肢'，'....’]
    # patientLabels:['鼻骨骨折 面部外伤 高血压','....']
    patientFeatures, patientLabels = dataProcess.load_data(filepath='../data/patientFeatures_test.csv')
    feature2id, id2feature, feature_vocab_size = dataProcess.build_vocab(patientFeatures, max_vocab_size=200)
    label2id, id2label, label_vocab_size = dataProcess.build_vocab(patientLabels)
    features = dataProcess.transform2ids(patientFeatures,feature2id,max_len)
    #print('features:', features)
    labels = dataProcess.labelEncoder(patientLabels,label2id)

    # STEP 3: 训练RL模型（包含RL模型的过程）
    train()




