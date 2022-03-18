import torch
import random
import numpy as np
from model_manager import Aggregator, Client
from data_manager import DatasetCreator
from utils.selection import select_clients
from utils.log_helper import record


def set_env(seed=5):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run():
    model_name = 'TwoNN'
    dataset_name = 'MNIST'
    partition_type = 'dirichlet'
    sample_method = 'random'
    rd = 50
    num_clients = 100
    sample_sz = 10
    batch_sz = 20
    epoch = 10
    lr = 0.01
    momentum = 0.9
    weight_decay = 0

    data_creator = DatasetCreator(num_clients, dataset_name, partition_type, alpha=0.1)

    client_ids = [_ for _ in range(1, num_clients+1)]
    clients = {}
    for i in range(num_clients):
        clients[i+1] = Client(model_name, num_classes=10)
        clients[i+1].init_data(i+1, data_creator, batch_sz)

    server = Aggregator(model_name, num_classes=10)
    server.init_data(data_creator, batch_sz)
    server.test()

    for cur_rd in range(1, rd+1):
        # select clients
        participants = select_clients(client_ids, sample_sz, sample_method)

        # server send global model to clients
        m_stat = server.get_model_state()
        for cid in participants:
            clients[cid].update_model(m_stat)

        # each selected client trains model
        for cid in participants:
            clients[cid].train(epoch, lr, momentum, weight_decay)

        # server gets updates from these clients
        for cid in participants:
            server.retrieve_update(cid, clients[cid].get_model_state(), clients[cid].get_training_data_len())

        # server updates global model
        server.update_model()

        # server tests global model
        acc, loss = server.test()

        print("Round {}, accuracy: {}%, loss: {}".format(cur_rd, round(acc*100, 2), round(loss, 2)))
        record(cur_rd, model_name, partition_type, sample_method, acc, loss)


if __name__ == '__main__':
    set_env()
    run()
