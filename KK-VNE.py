from substrate import Substrate
from topology_maker import simulate_events_one
from analysis import Analysis
import time


def kk_vne():
    # Step1: read substrate network and virtual network request file
    network_substrate_path = 'networks/'
    # network_filename = 'sub100-570.txt'
    network_filename = 'subts.txt'
    # 构建物理网络
    sub = Substrate(network_substrate_path, network_filename)
    # 构建虚拟网络映射请求
    event_queue = simulate_events_one('VNRequest/', 2000)

    # Step2: choose mapping algorithm
    algorithm = 'kkDQN'
    # 训练的轮次
    arg = 10

    # Step3: deal virtual network request event
    # 处理虚拟网络映射请求
    start = time.time()
    sub.handle(event_queue, algorithm, arg)
    time_cost = time.time() - start
    print("cost time %d" % time_cost)

    # Step4: ouput mapping result file
    analysis = Analysis()
    analysis.save_result(sub, '%s-VNE-%s-cacln-1.txt' % (algorithm, arg))


if __name__ == '__main__':
    kk_vne()
