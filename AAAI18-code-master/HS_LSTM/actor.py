import tensorflow as tf
import numpy as np
import tflearn

class ActorNetwork(object):
    """
    action network
    use the state
    sample the action
    """

    def __init__(self, sess, dim, optimizer, learning_rate, tau, num_critic_vars):
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
        print (self.optimizer)
        self.num_other_variables = len(tf.trainable_variables())

        #actor network(updating)
        self.input_l, self.input_d, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()[self.num_other_variables:]

        #actor network(delayed updating)
        self.target_input_l, self.target_input_d, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[self.num_other_variables + len(self.network_params):]

        #delayed updaing actor network
        #更新target网络的参数
        self.update_target_network_params = \
                [self.target_network_params[i].assign(\
                tf.multiply(self.network_params[i], self.tau) +\
                tf.multiply(self.target_network_params[i], 1 - self.tau))\
                for i in range(len(self.target_network_params))]

        #将target网络的参数赋值给network网络
        self.assign_active_network_params = \
                [self.network_params[i].assign(\
                self.target_network_params[i]) for i in range(len(self.network_params))]

        #gradient provided by critic network
        self.action_gradient = tf.placeholder(tf.float32, [2])
        # logpi(at|st)
        self.log_target_scaled_out = tf.log(self.target_scaled_out)

        self.actor_gradients = tf.gradients(self.log_target_scaled_out, self.target_network_params, self.action_gradient)

        self.grads = [tf.placeholder(tf.float32, [600, 1]),
                        tf.placeholder(tf.float32, [1,]),
                        tf.placeholder(tf.float32, [600, 1])]
        self.optimize = self.optimizer.apply_gradients(zip(self.grads, self.network_params[:-1]), global_step=self.global_step)

    ''''
    输入状态值 返回action的概率
    '''
    def create_actor_network(self):
        # 状态为[ct-1;ht-1;ct;ht]
        input_l = tf.placeholder(tf.float32, shape=[1, self.dim*2])
        input_d = tf.placeholder(tf.float32, shape=[1, self.dim*2])
        
        t1 = tflearn.fully_connected(input_l, 1)
        t2 = tflearn.fully_connected(input_d, 1)

        scaled_out = tflearn.activation(\
                tf.matmul(input_l,t1.W) + tf.matmul(input_d,t2.W) + t1.b,\
                activation = 'sigmoid')
        
        scaled_out = tf.stack([1.0 - scaled_out[0][0], scaled_out[0][0]])
        return input_l, input_d, scaled_out
    
    def train(self, grad):
        self.sess.run(self.optimize, feed_dict={
            self.grads[0]: grad[0],
            self.grads[1]: grad[1],
            self.grads[2]: grad[2]})

    def predict_target(self, input_l, input_d):
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
