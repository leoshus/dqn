import tensorflow as tf
import numpy as np
from kk.kk_mdp import KKEnv
import copy
from collections import deque
import random
import os


class NoisyDQN:
    def __init__(self, sub, n_actions, n_features, num_epoch):
        self.sub = sub
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = 0.01
        self.reward_decay = 0.9
        self.e_greedy = 0.9
        self.replace_target_iter = 300
        self.memory_size = 2000
        self.batch_size = 64
        self.e_greedy_increment = None
        self.episodes = num_epoch

        self.learning_step_counter = 0
        self.epsilon = 0 if self.e_greedy_increment is not None else self.e_greedy
        self.memory = deque(maxlen=self.memory_size)
        self.create_model()
        self.cost_his = []
        self.result_dir = 'results/'

    def create_model(self):
        '''建立预测模型和target模型'''
        dimension = 32
        # ------------------ build evaluate_net ------------------
        s = tf.keras.Input([None, self.n_features], name='s')
        # 预测模型
        x = tf.keras.layers.Dense(dimension, activation=tf.keras.activations.relu, name='l1')(s)

        x = NoisyDense(dimension, input_dim=x.get_shape().as_list()[2])(x)
        x = tf.keras.layers.Dense(1, name='l2')(x)
        self.eval_net = tf.keras.Model(inputs=s, outputs=x)
        # 损失计算函数
        self.loss = tf.keras.losses.MeanSquaredError()
        # 梯度下降方法
        self._train_op = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        # ------------------ build target_net ------------------
        s_ = tf.keras.Input([None, self.n_features], name='s_')
        # target模型
        x = tf.keras.layers.Dense(dimension, activation=tf.keras.activations.relu, name='l1')(s_)
        x = NoisyDense(dimension, input_dim=x.get_shape().as_list()[2])(x)
        x = tf.keras.layers.Dense(1, name='l2')(x)
        self.target_net = tf.keras.Model(inputs=s_, outputs=x)

    def save_model(self, fn):
        # save model to file, give file name with .h5 extension
        self.eval_net.save(fn)

    def load_model(self, fn):
        self.eval_net = tf.keras.models.load_model(fn, compile=False, custom_objects={"NoisyDense": NoisyDense})

        # self.target_net.set_weights(self.eval_net.get_weights())

    def target_update(self):
        weights = self.eval_net.get_weights()
        target_weights = self.target_net.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_net.set_weights(target_weights)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        A = tf.one_hot(a, self.n_actions).numpy()
        # transition = np.hstack((s, A, A * r, s_))
        # transition = np.hstack((s, s_))
        # index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        self.memory.append([s, a, r, s_])
        self.memory_counter += 1

    def choose_action(self, observation, sub, current_node_cpu, acts):
        observation = observation[np.newaxis, :]
        actions_value = self.eval_net(observation).numpy()
        candidate_action = []
        candidate_score = []
        index = 0
        for score in actions_value[0]:
            # print(index, score[0])
            if index not in acts and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
                candidate_action.append(index)
                candidate_score.append(score[0])
            index += 1
        if len(candidate_action) == 0:
            return -1
        else:
            candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            # 选择动作
            action = np.random.choice(candidate_action, p=candidate_prob)
            return action

    def choose_predict_action(self, observation, sub, current_node_cpu, acts):
        observation = observation[np.newaxis, :]
        actions_value = self.eval_net(observation).numpy()
        candidate_action = []
        candidate_score = []
        index = 0
        for score in actions_value[0]:
            # print(index, score[0])
            if index not in acts and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
                candidate_action.append(index)
                candidate_score.append(score[0])
            index += 1
        if len(candidate_action) == 0:
            return -1
        else:
            candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            # 选择动作
            action = np.random.choice(candidate_action, p=candidate_prob)
            return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        if self.learning_step_counter % self.replace_target_iter == 0:
            self.target_update()

        samples = random.sample(self.memory, self.batch_size)
        s, a, r, s_ = map(np.asarray, zip(*samples))
        batch_s = np.array(s).reshape(self.batch_size, self.n_actions, -1)
        batch_s_ = np.array(s_).reshape(self.batch_size, self.n_actions, -1)

        with tf.GradientTape() as tape:
            q_next = self.target_net(batch_s_)
            q_eval = self.eval_net(batch_s)
            # print(q_next, q_eval)

            q_target = q_eval.numpy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = np.array(a).reshape(self.batch_size, -1)
            reward = np.array(r).reshape(self.batch_size, -1)
            q_target[batch_index, eval_act_index] = reward + self.reward_decay * np.max(q_next, axis=1)
            self.cost = self.loss(y_true=q_target, y_pred=q_eval)
        gradients = tape.gradient(self.cost, self.eval_net.trainable_variables)
        self._train_op.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy else self.e_greedy
        self.learning_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost(loss)')
        plt.xlabel('training steps')
        plt.savefig(self.result_dir + 'loss.png')
        plt.show()

    def train(self, training_set):
        for episode in range(self.episodes):
            # 每轮训练开始前，都需要重置底层网络和相关的强化学习环境
            sub_copy = copy.deepcopy(self.sub)
            # 构建环境
            env = KKEnv(self.sub.net)
            # 记录已经处理的虚拟网络请求数量
            counter = 0
            for req in training_set:
                env.render()
                # 当前待映射的虚拟网络请求ID
                req_id = req.graph['id']
                print("\n Handling req %s ..." % req_id)

                if req.graph['type'] == 0:
                    counter += 1
                    sub_copy.total_arrived = counter
                    # 向环境传入当前的待映射虚拟网络
                    env.set_vnr(req)
                    # 获得底层网络的状态
                    observation = env.reset()
                    node_map = {}
                    acts = []
                    queue = deque(maxlen=req.number_of_nodes())
                    for vn_id in range(req.number_of_nodes()):
                        sn_id = self.choose_action(observation, sub_copy.net, req.nodes[vn_id]['cpu'], acts)
                        if sn_id == -1:
                            break
                        else:
                            acts.append(sn_id)
                            observation_, reward, done, info = env.step(sn_id)
                            node_map.update({vn_id: sn_id})
                            queue.append([observation, sn_id, observation_])
                            observation = observation_
                    if len(node_map) == req.number_of_nodes():
                        # if step > 200 and step % 5 == 0:
                        print("req %s mapping success" % req_id)
                        reward, link_map = self.calculate_reward(sub_copy, req, node_map)
                        sub_copy.mapped_info.update({req.graph['id']: (node_map, link_map)})
                        sub_copy.change_resource(req, 'allocate')
                        while queue:
                            observation, sn_id, observation_ = queue.popleft()
                            self.store_transition(observation, sn_id, reward, observation)
                        self.learn()
                    else:
                        print("mapping Failure!")
                if req.graph['type'] == 1:
                    print("\tIt's time is out, release the occupied resources")
                    if req_id in sub_copy.mapped_info.keys():
                        sub_copy.change_resource(req, 'release')
                env.set_sub(sub_copy.net)

    def calculate_reward(self, sub, req, node_map):
        link_map = sub.link_mapping(req, node_map)
        if len(link_map) == req.number_of_edges():
            requested, occupied = 0, 0

            # node resource
            for vn_id, sn_id in node_map.items():
                node_resource = req.nodes[vn_id]['cpu']
                occupied += node_resource
                requested += node_resource

            # link resource
            for vl, path in link_map.items():
                link_resource = req[vl[0]][vl[1]]['bw']
                requested += link_resource
                occupied += link_resource * (len(path) - 1)

            reward = requested / occupied

            return reward, link_map
        else:
            return -1, link_map

    def run(self, sub, req):
        self.load_model('kk/model/NoisyDQN.h5')
        node_map = {}
        env = KKEnv(sub.net)
        env.set_vnr(req)
        observation = env.reset()
        acts = []
        for vn_id in range(req.number_of_nodes()):
            sn_id = self.choose_predict_action(observation, sub.net, req.nodes[vn_id]['cpu'], acts)
            if sn_id == -1:
                return
            else:
                acts.append(sn_id)
                observation, _, done, info = env.step(sn_id)
                node_map.update({vn_id: sn_id})
        return node_map


class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, std_init=0.5):
        super(NoisyDense, self).__init__()
        self.units = units
        self.std_init = std_init
        self.input_dim = input_dim
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))
        self.weight_mu = tf.Variable(name="weight_mu",
                                     initial_value=mu_initializer(shape=(input_dim, units), dtype=tf.float32),
                                     trainable=True)
        self.weight_sigma = tf.Variable(name="weight_sigma",
                                        initial_value=sigma_initializer(shape=(input_dim, units), dtype=tf.float32),
                                        trainable=True)
        self.bias_mu = tf.Variable(name="bias_mu", initial_value=mu_initializer(shape=(units,), dtype=tf.float32),
                                   trainable=True)
        self.bias_sigma = tf.Variable(name="bias_sigma",
                                      initial_value=sigma_initializer(shape=(units,), dtype=tf.float32),
                                      trainable=True)

    # @tf.function
    def call(self, inputs):
        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out

    def get_config(self):
        config = {
            'units': self.units,
            'input_dim': self.input_dim,
            'std_init': self.std_init,
        }
        base_config = super(NoisyDense, self).get_config()
        return dict(list(config.items()))
        # config = super().get_config().copy()
        # config.update({
        #     'units': self.units,
        #     'input_dim': self.input_dim,
        #     'std_init': self.std_init,
        #     # 'weight_mu': self.weight_mu,
        #     # 'weight_sigma': self.weight_sigma,
        #     # 'bias_mu': self.bias_mu,
        #     # 'bias_sigma': self.bias_sigma,
        #     # 'kernel': self.kernel,
        #     # 'bias': self.bias,
        #     # 'weights_eps': self.weights_eps,
        #     # 'bias_eps': self.bias_eps,
        # })
        # return config
