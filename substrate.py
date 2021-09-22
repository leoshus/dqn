from topology_maker import extract_network
from evalution import Evaluation
from configure import configure
import networkx as nx
from DeepQN import DQN


# 网络构建
class Substrate:
    def __init__(self, path, filename):
        # 构建的网络
        self.net = extract_network(path, filename)
        # 智能体/训练的模型
        self.agent = None
        # 映射信息
        self.mapped_info = {}
        # 指标评估
        self.evaluation = Evaluation(graph=self.net)
        self.no_solution = False
        pass

    def set_topology(self, graph):
        self.net = graph

    def handle(self, queue, algorithm, arg=0):
        for req in queue:
            # the id of current request
            req_id = req.graph['id']
            if req.graph['type'] == 0:
                """a request which is newly arrived"""
                print("\nTry to map request:%s" % req_id)
                # 虚拟网络映射请求 映射到物理网络
                if self.mapping(req, algorithm, arg):
                    print("Success!")
            if req.graph['type'] == 1:
                """a request which is ready to leave """
                if req_id in self.mapped_info.keys():
                    print("\nRelease the resources which are occupied by request %s" % req_id)
                    self.change_resource(req, 'release')

    def mapping(self, vnr, algorithm, arg):
        """tow phrase:node mapping and link mapping"""
        # 评估指标记录映射请求到达的数量
        self.evaluation.total_arrived += 1
        # mapping virtual nodes 虚拟网络映射请求 节点映射
        node_map = self.node_mapping(vnr, algorithm, arg)
        if len(node_map) == vnr.number_of_nodes():
            # mapping virtual links
            link_map = self.link_mapping(vnr, node_map)
            if len(link_map) == vnr.number_of_edges():
                self.mapped_info.update({vnr.graph['id']: (node_map, link_map)})
                self.change_resource(vnr, 'allocate')
                print("Success!")
                return True
            else:
                print("Failed to map all links!")
                return False
        else:
            print("Failed to map all nodes!")
            return False

    def node_mapping(self, vnr, algorithm, arg):
        """求解节点映射问题"""
        print("node mapping ...")
        node_map = {}
        # 如果刚开始映射,那么需要对所选用的算法进行配置
        if self.agent is None:
            self.agent = configure(self, algorithm, arg)
        node_map = self.agent.run(self, vnr)
        # 返回节点映射集合
        return node_map

    def link_mapping(self, vnr, node_map):
        """求解链路映射问题"""
        link_map = {}
        for vLink in vnr.edges:
            vn_from = vLink[0]
            vn_to = vLink[1]
            sn_from = node_map[vn_from]
            sn_to = node_map[vn_to]
            if nx.has_path(self.net, source=sn_from, target=sn_to):
                for path in nx.all_shortest_paths(self.net, sn_from, sn_to):
                    if self.get_path_capacity(path) >= vnr[vn_from][vn_to]['bw']:
                        link_map.update({vLink: path})
                        break
                    else:
                        continue
        # 返回链路映射的集合
        return link_map

    def get_path_capacity(self, path):
        """找到一条路径中带宽资源最小的链路并返回器带宽资源值"""
        bandwidth = 1000
        head = path[0]
        for tail in path[1:]:
            if self.net[head][tail]['bw_remain'] <= bandwidth:
                bandwidth = self.net[head][tail]['bw_remain']
            head = tail
        return bandwidth

    def change_resource(self, req, instruction):
        """分配或释放节点和链路资源"""
        # 读取该虚拟网络请求的映射信息
        req_id = req.graph['id']
        node_map = self.mapped_info[req_id][0]
        link_map = self.mapped_info[req_id][1]
        factor = -1
        if instruction == 'release':
            factor = 1
        # 分配或释放节点资源
        for v_id, s_id in node_map.items():
            self.net.nodes[s_id]['cpu_remain'] += factor * req.nodes[v_id]['cpu']
        # 分配或释放链路资源
        for vl, path in link_map.items():
            link_resource = req[vl[0]][vl[1]]['bw']
            start = path[0]
            for end in path[1:]:
                self.net[start][end]['bw_remain'] += factor * link_resource
                start = end
        if instruction == 'allocate':
            # 增加实验结果
            self.evaluation.add(req, link_map)
        if instruction == 'release':
            # 移出相应的映射信息
            self.mapped_info.pop(req_id)
