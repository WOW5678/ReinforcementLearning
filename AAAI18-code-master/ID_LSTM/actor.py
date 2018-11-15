import tensorflow as tf
import numpy as np
import tflearn

class ActorNetwork(object):
    """
    action network
    use the state
    sample the action
    """

    def __init__(self, sess, dim, optimizer, learning_rate, tau):
        self.global_step = tf.Variable(0, trainable=False, name="ActorStep")
        self.sess = sess
        self.dim = dim
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.95, staircase=True)
        self.tau = tau
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        self.num_other_variables = len(tf.trainable_variables())

        #actor network(updating)
        self.input_l, self.input_d, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()[self.num_other_variables:]
        print('actor network_params:',self.network_params)

        #actor network(delayed updating)
        self.target_input_l, self.target_input_d, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[self.num_other_variables + len(self.network_params):]
        print('actor target_network_params:',self.target_network_params)
        #delayed updaing actor network
        #将self.network_params中的参数经过一定变化之后赋值给self.target_network_paras
        # self.update_target_network_paras是个操作

        self.update_target_network_params = \
                [self.target_network_params[i].assign(\
                tf.multiply(self.network_params[i], self.tau) +\
                tf.multiply(self.target_network_params[i], 1 - self.tau))\
                for i in range(len(self.target_network_params))]

        #将self.target_network_params中的参数赋值给self.network_params
        # self.assign_active_network_params是个操作
        self.assign_active_network_params = \
                [self.network_params[i].assign(\
                self.target_network_params[i]) for i in range(len(self.network_params))]


        #gradient provided by critic network
        # 从critic 网络中产生的 作为reward 更新actor network的参数
        self.action_gradient = tf.placeholder(tf.float32, [2])
        # 对概率值取log操作
        self.log_target_scaled_out = tf.log(self.target_scaled_out)

        # self.action_gradient被认为是常量 不求他们的梯度
        # self.actor_gradients有四个元素，但最后一个元素为None,因为我们设置了stop_gradients=action_gradients
        #logpi(at|st)的梯度
        self.actor_gradients = tf.gradients(self.log_target_scaled_out, self.target_network_params, stop_gradients=self.action_gradient)
        print ('self.actor_gradients:',self.actor_gradients)

        ######################################################################
        #这个grads也是通过critic network 返回的reward对网络求导得到的
        #用来进一步更新network_params
        self.grads = [tf.placeholder(tf.float32, [600,1]), 
                        tf.placeholder(tf.float32, [1,]),
                        tf.placeholder(tf.float32, [300, 1])]

        # 为什么要加上个[:-1]呢？？？
        #因为self.network_params是一个包含四个元素的列表，最后一个是[1]
        #而这里的grads只有三个元素
        self.optimize = self.optimizer.apply_gradients(zip(self.grads, self.network_params[:-1]), global_step=self.global_step)
    '''
    actor network就是输入不同的状态 产生该状态下不同action的概率
    '''
    def create_actor_network(self):
        #input_1:是当前的state,input_d是当前输入的字符的表示
        #这两者共同组成了当前时刻的actor的state
        #input_1:其实就是：[ct-1；ht-1] input_d：xt
        input_l = tf.placeholder(tf.float32, shape=[1, self.dim*2])
        input_d = tf.placeholder(tf.float32, shape=[1, self.dim])
        
        t1 = tflearn.fully_connected(input_l, 1)
        t2 = tflearn.fully_connected(input_d, 1)

        # scaled_out是action=1的概率
        scaled_out = tflearn.activation(\
                tf.matmul(input_l,t1.W) + tf.matmul(input_d,t2.W) + t1.b,\
                activation = 'sigmoid')
        #将这个概率的值进行限定
        # s_out是个float值
        s_out = tf.clip_by_value(scaled_out[0][0], 1e-5, 1 - 1e-5)

        #scaled_out:[2,] 是个行向量 表示action=0的概率和action=1的概率
        scaled_out = tf.stack([1.0 - s_out, s_out])
        return input_l, input_d, scaled_out
    
    def train(self, grad):
        self.sess.run(self.optimize, feed_dict={
            self.grads[0]: grad[0], 
            self.grads[1]: grad[1],
            self.grads[2]: grad[2]})

    def predict_target(self, input_l, input_d):
        # self.target_scaled_out是一个行向量，两个元素，分别为action为0的概率和为1的概率
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_input_l: input_l,
            self.target_input_d: input_d})
    
    def get_gradient(self, input_l, input_d, a_gradient):
        return self.sess.run(self.actor_gradients[:-1], feed_dict={
            self.target_input_l: input_l,
            self.target_input_d: input_d,
            self.action_gradient: a_gradient})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)
