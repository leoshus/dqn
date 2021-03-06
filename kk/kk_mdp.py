import gym
from gym import spaces
import copy
import networkx as nx
import numpy as np


class KKEnv(gym.Env):

    def __init__(self, sub, n_feature=4):
        self.count = -1
        self.n_feature = n_feature
        self.n_action = sub.number_of_nodes()
        self.sub = copy.deepcopy(sub)
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_action, self.n_feature), dtype=np.float32)
        self.state = None
        self.actions = []
        self.degree = []
        for i in nx.degree_centrality(sub).values():
            self.degree.append(i)
        self.vnr = None
        self.cln = []
        for j in nx.closeness_centrality(sub).values():
            self.cln.append(j)

    def set_sub(self, sub):
        self.sub = copy.deepcopy(sub)

    def set_vnr(self, vnr):
        self.vnr = vnr

    def step(self, action):
        self.actions.append(action)
        self.count += 1
        cpu_remain, bw_all_remain, avg_dst = [], [], []
        for u in range(self.n_action):
            adjacent_bw = calculate_adjacent_bw(self.sub, u, 'bw_remain')
            if u == action:
                self.sub.nodes[action]['cpu_remain'] -= self.vnr.nodes[self.count]['cpu']
                adjacent_bw -= calculate_adjacent_bw(self.vnr, self.count)
            cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            bw_all_remain.append(adjacent_bw)

            sum_dst = 0
            for v in self.actions:
                sum_dst += nx.shortest_path_length(self.sub, source=u, target=v)
            sum_dst /= (len(self.actions) + 1)
            avg_dst.append(sum_dst)
        cpu_remain = (cpu_remain - np.min(cpu_remain)) / (np.max(cpu_remain) - np.min(cpu_remain))
        bw_all_remain = (bw_all_remain - np.min(bw_all_remain)) / (np.max(bw_all_remain) - np.min(bw_all_remain))
        avg_dst = (avg_dst - np.min(avg_dst)) / (np.max(avg_dst) - np.min(avg_dst))
        k_feature = self.cal_nsf()
        k_feature = (k_feature - np.min(k_feature)) / (np.max(k_feature) - np.min(k_feature))
        self.state = (cpu_remain,
                      bw_all_remain,
                      self.degree,
                      self.cln,
                      avg_dst,
                      k_feature)
        return np.vstack(self.state).transpose(), 0.0, False, {}

    def reset(self):
        """???????????????????????????????????????"""
        self.count = -1
        self.actions = []
        cpu_remain, bw_all_remain = [], []
        for u in range(self.n_action):
            cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            bw_all_remain.append(calculate_adjacent_bw(self.sub, u, 'bw_remain'))
        cpu_remain = (cpu_remain - np.min(cpu_remain)) / (np.max(cpu_remain) - np.min(cpu_remain))
        bw_all_remain = (bw_all_remain - np.min(bw_all_remain)) / (np.max(bw_all_remain) - np.min(bw_all_remain))
        avg_dst = np.zeros(self.n_action).tolist()
        k_feature = self.cal_nsf()
        k_feature = (k_feature - np.min(k_feature)) / (np.max(k_feature) - np.min(k_feature))
        self.state = (cpu_remain,
                      bw_all_remain,
                      self.degree,
                      self.cln,
                      avg_dst,
                      k_feature)
        return np.vstack(self.state).transpose()

    def cal_nsf(self):
        g = copy.deepcopy(self.sub)
        importance_dict = [0 for i in range(0, g.number_of_nodes())]
        s = 0
        # ??????????????????????????????????????????
        while (g.number_of_nodes() > 0):
            # ????????????ks?????????
            temp = []
            # for k,i in zip(list(g.node),range(0,len(g.node))):
            # for k in range(g.number_of_nodes()):
            for k in list(g.nodes()):
                node_nei = list(g.neighbors(k))
                v = len(node_nei)
                if v <= s:
                    # lsum = k??????????????????????????????
                    # ??????????????????
                    # node_nei = list(g.neighbors(k))
                    r_band = 0
                    # ??????????????????????????????
                    for n_n in node_nei:
                        r_band = r_band + g.get_edge_data(k, n_n)['bw_remain']
                    temp.append("1")
                    shuzhi = (s * r_band) ** 0.5 * g.nodes[k]['cpu_remain']
                    importance_dict[k] = shuzhi
                    g.remove_node(k)
            # ????????????ks1???????????????????????????????????????s????????????????????????s+1
            if len(temp) == 0:
                s += 1
        return importance_dict

    def render(self, mode='human'):
        pass


def calculate_adjacent_bw(graph, u, kind='bw'):
    """???????????????????????????????????????????????????????????????????????????????????????????????????????????????kind?????????bw-remain"""
    bw_sum = 0
    for v in graph.neighbors(u):
        bw_sum += graph[u][v][kind]
    return bw_sum
