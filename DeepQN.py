import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, models, layers, optimizers
from tensorflow.keras.optimizers import RMSprop
from kk.kk_mdp import KKEnv
import copy
from collections import deque
import random


class DQN:
    def __init__(self, sub, n_actions, n_features, num_epoch):
        self.sub = sub
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = 0.01
        self.reward_decay = 0.9
        self.e_greedy = 0.9
        self.replace_target_iter = 300
        self.memory_size = 500
        self.batch_size = 32
        self.e_greedy_increment = None
        self.episodes = num_epoch

        self.learning_step_counter = 0
        self.epsilon = 0 if self.e_greedy_increment is not None else self.e_greedy
        # self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.memory = deque(maxlen=self.memory_size)
        self.create_model()
        self.cost_his = []

    def create_model(self):
        '''建立预测模型和target模型'''
        # ------------------ build evaluate_net ------------------
        s = tf.keras.Input([None, self.n_features], name='s')
        # 预测模型
        x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='l1')(s)
        x = tf.keras.layers.Dense(1, name='l2')(x)
        self.eval_net = tf.keras.Model(inputs=s, outputs=x)
        # 损失计算函数
        self.loss = tf.keras.losses.MeanSquaredError()
        # 梯度下降方法
        self._train_op = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        # ------------------ build target_net ------------------
        s_ = tf.keras.Input([None, self.n_features], name='s_')
        # target模型
        x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, name='l1')(s_)
        x = tf.keras.layers.Dense(1, name='l2')(x)
        self.target_net = tf.keras.Model(inputs=s_, outputs=x)

    def save_model(self, fn):
        # save model to file, give file name with .h5 extension
        self.eval_net.save(fn)

    def load_model(self, fn):
        self.eval_net = tf.keras.models.load_model(fn)

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

        if np.random.uniform() < self.epsilon:
            # print(observation.shape)
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
        else:
            action = np.random.randint(0, self.n_actions)
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
        # batch_eval = self.target_net(batch_s)
        # q_future = self.eval_net(batch_s_)
        # print(batch_eval, q_future)
        # batch_eval[range(self.batch_size), a] = r + q_future * self.reward_decay
        # history = self.eval_net(batch_s, batch_eval)

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
        # q_target = q_eval.copy()
        #
        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features + 1]
        #
        # q_target[batch_index, eval_act_index] = reward + self.reward_decay * np.max(q_next, axis=1)
        #
        # if self.learning_step_counter % self.replace_target_iter == 0:
        #     for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
        #         target_layer.set_weights(eval_layer.get_weights())
        #     print('\n target_param_replaced\n')
        # self.cost = self.eval_model.train_on_batch(batch_memory[:, :self.n_features], q_target)
        # self.cost_his.append(self.cost)
        # self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy else self.e_greedy
        # self.learning_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def train(self, training_set):
        step = 0
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
                    for vn_id in range(req.number_of_nodes()):
                        sn_id = self.choose_action(observation, sub_copy.net, req.nodes[vn_id]['cpu'], acts)
                        if sn_id == -1:
                            break
                        else:
                            acts.append(sn_id)
                            observation_, reward, done, info = env.step(sn_id)
                            self.store_transition(observation, sn_id, reward, observation_)
                            node_map.update({vn_id: sn_id})
                            observation = observation_
                    if len(node_map) == req.number_of_nodes():
                        # if step > 200 and step % 5 == 0:
                        print("req %s mapping success" % req_id)
                        self.learn()
                    else:
                        print("mapping Failure!")
                if req.graph['type'] == 1:
                    print("\tIt's time is out, release the occupied resources")
                    if req_id in sub_copy.mapped_info.keys():
                        sub_copy.change_resource(req, 'release')
                env.set_sub(sub_copy.net)
        env.destory()

    def run(self):
        pass


if __name__ == "__main__":
    RL = DQN()

    RL.plot_cost()
