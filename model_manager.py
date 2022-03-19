import csv

from tqdm import tqdm
import gc
import copy
import math
import random
import numpy as np
import torch
import torchvision
from utils.models import *
from utils.algrithms import FedAvg


class Aggregator(object):
    """A server"""

    def __init__(self, model_name, num_classes, num_clients):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda')
        self.model = self.__init_model()
        self.accuracy = 0
        self.loss = 0
        self.collected_updates = None
        self.absolute_weights = None
        self.avg_shapley_values = {}
        self.last_involved_round = {}
        self.num_join = {}
        self.test_loader = None
        self.unexplored = [_ for _ in range(1, num_clients+1)]
        self.explored = []
        self.explore_ratio = 0.9
        self.cur_rd = 1

    def __init_model(self):
        if self.model_name == 'TwoNN':
            return TwoNN(num_classes=self.num_classes)
        elif self.model_name == 'CNN':
            return CNN(num_classes=self.num_classes)
        elif self.model_name == 'shufflenet_v2_x2_0':
            return torchvision.models.shufflenet_v2_x2_0(num_classes=self.num_classes)
        elif self.model_name == 'mobilenet_v2':
            return torchvision.models.mobilenet_v2(num_classes=self.num_classes)
        else:
            print("ERROR: unknown model type.")
            exit(-1)

    def init_data(self, data_creator, batch_sz, time_out=0, num_workers=0):
        self.test_loader = data_creator.get_loader(batch_sz=batch_sz, is_test=True, time_out=time_out,
                                                   num_workers=num_workers)
        self.accuracy, self.loss = self.test(self.model)

    def select_clients(self, sample_sz, sample_method='random'):
        if sample_method == 'random':
            return random.sample(self.unexplored, sample_sz)
        elif sample_method == 'bandit':
            explore_num = min(len(self.unexplored), int(self.explore_ratio * sample_sz + 0.5))
            exploit_num = sample_sz - explore_num

            if len(self.explored) < exploit_num:
                participants = random.sample(self.unexplored, sample_sz)
            else:
                utility = {}
                for cid in self.explored:
                    L = self.last_involved_round[cid]
                    utility[cid] = self.avg_shapley_values[cid] + math.sqrt(0.1 * math.log(self.cur_rd, 10) / L)

                unexplored_participants = random.sample(self.unexplored, explore_num)
                explored_participants = sorted(utility, key=utility.get, reverse=True)[:exploit_num]
                participants = unexplored_participants + explored_participants

            return participants
        else:
            print("ERROR: unknown sample method.")

    def get_test_data_len(self):
        if self.test_loader is None:
            print("ERROR: not init_data yet.")
            exit(-1)
        return len(self.test_loader.dataset)

    def test(self, model):
        # print("### Testing...")
        model = model.to(device=self.device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss().cuda()
        accuracy = loss = 0

        for (X, y) in self.test_loader:
            X = X.to(device=self.device)
            y = y.to(device=self.device)
            output = model(X)
            loss += criterion(output, y).item()
            predicted = output.argmax(dim=1, keepdim=True)
            accuracy += predicted.eq(y.view_as(predicted)).sum().item()

        accuracy /= len(self.test_loader.dataset)
        loss /= len(self.test_loader)
        # print("Model {} test accuracy {}%, loss {}".format(self.model_name, round(accuracy*100, 2), round(loss, 2)))

        torch.cuda.empty_cache()
        model.to('cpu')

        return accuracy, loss

    def retrieve_update(self, client_id, update, weight):
        """Retrieve udpate from a client"""

        if self.collected_updates is None:
            self.collected_updates = {}
            self.absolute_weights = {}

        self.collected_updates[client_id] = update
        self.absolute_weights[client_id] = weight

    def update_shapley_values(self, n):
        """
        Approximate shapley values by monte-carlo simulation.
        DO NOT invoke update_model if already invoke this function
        """

        shapley_values = {}
        values = {}
        weight = 1. / n
        precede_cmb = ()
        values[precede_cmb] = self.accuracy
        # values[precede_cmb] = self.loss

        all_participants = tuple(sorted(self.collected_updates.keys()))
        self.__get_value(all_participants, do_update=True)
        values[all_participants] = self.accuracy
        # values[all_participants] = self.loss

        for cid in all_participants:
            shapley_values[cid] = 0

        for _ in tqdm(range(n), colour='blue'):
            permutation = np.random.permutation(list(self.collected_updates.keys()))
            precede_cmb = ()
            for cid in permutation:
                cur_cmb = tuple(sorted(precede_cmb + (cid,)))
                if cur_cmb not in values:
                    values[cur_cmb], _ = self.__get_value(cur_cmb)
                    # _, values[cur_cmb] = self.__get_value(cur_cmb)
                shapley_values[cid] += weight * (values[cur_cmb] - values[precede_cmb])
                # shapley_values[cid] += weight * (values[precede_cmb] - values[cur_cmb])
                precede_cmb = cur_cmb

        for cid in all_participants:
            if cid not in self.last_involved_round:
                self.avg_shapley_values[cid] = shapley_values[cid]
                self.num_join[cid] = 1
                self.last_involved_round[cid] = self.cur_rd
                self.explored.append(cid)
                self.unexplored.remove(cid)
            else:
                self.last_involved_round[cid] = self.cur_rd
                if self.num_join[cid] == 10:
                    del self.last_involved_round[cid]
                    self.explored.remove(cid)
                    self.unexplored.append(cid)
                    del self.avg_shapley_values[cid]
                    del self.num_join[cid]
                else:
                    self.avg_shapley_values[cid] = (self.avg_shapley_values[cid] * self.num_join[cid] + shapley_values[
                        cid]) / (self.num_join[cid] + 1)
                    self.num_join[cid] += 1

        self.cur_rd += 1
        self.explore_ratio = max(self.explore_ratio*0.98, 0.2)
        self.collected_updates = None
        self.absolute_weights = None
        gc.collect()

    def update_model(self):
        """Update model when has collected enough updates"""

        client_ids = list(self.collected_updates.keys())
        coefficients = {}
        total_weights = sum(self.absolute_weights.values())
        for cid in client_ids:
            coefficients[cid] = self.absolute_weights[cid] / total_weights

        model_state = FedAvg(client_ids, self.collected_updates, coefficients)
        self.model.load_state_dict(model_state)
        self.collected_updates = None
        self.absolute_weights = None
        gc.collect()

    def get_model_state(self):
        return copy.deepcopy(self.model.state_dict())

    def __get_value(self, combination, do_update=False):
        """Help update_shapley_value"""

        coefficients = {}
        total_weights = 0
        for cid in combination:
            total_weights += self.absolute_weights[cid]
        for cid in combination:
            coefficients[cid] = self.absolute_weights[cid] / total_weights

        model_state = FedAvg(combination, self.collected_updates, coefficients)
        model = self.__init_model()
        model.load_state_dict(model_state)

        if do_update:
            self.model.load_state_dict(model_state)

        acc, loss = self.test(model)
        if do_update:
            self.accuracy = acc
            self.loss = loss

        del model
        gc.collect()

        return acc, loss


class Client(object):
    """A client"""

    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda')
        self.model = self.__init_model()
        self.accuracy = 0
        self.loss = 0
        # self.absolute_weight = None
        self.training_loader = None
        self.test_loader = None

    def __init_model(self):
        if self.model_name == 'TwoNN':
            return TwoNN(num_classes=self.num_classes)
        elif self.model_name == 'CNN':
            return CNN(num_classes=self.num_classes)
        elif self.model_name == 'shufflenet_v2_x2_0':
            return torchvision.models.shufflenet_v2_x2_0(num_classes=self.num_classes)
        elif self.model_name == 'mobilenet_v2':
            return torchvision.models.mobilenet_v2(num_classes=self.num_classes)
        else:
            print("ERROR: unknown model type.")
            exit(-1)

    def init_data(self, client_id, data_creator, batch_sz, time_out=0, num_workers=0):
        self.training_loader = data_creator.get_loader(client_id, batch_sz=batch_sz, is_test=False,
                                                       time_out=time_out, num_workers=num_workers)
        self.test_loader = data_creator.get_loader(batch_sz=batch_sz, is_test=True, time_out=time_out,
                                                   num_workers=num_workers)

    def get_training_data_len(self):
        if self.training_loader is None:
            print("ERROR: not init_data yet.")
            exit(-1)
        return len(self.training_loader.dataset)

    def get_test_data_len(self):
        if self.test_loader is None:
            print("ERROR: not init_data yet.")
            exit(-1)
        return len(self.test_loader.dataset)

    def train(self, epoch, lr=1e-2, momentum=0.9, weight_decay=4e-5):
        print("### Training...")
        self.model = self.model.to(device=self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss().cuda()

        step = 0
        with tqdm(total=100, colour='green') as pbar:
            while step < epoch:
                step += 1
                self.model.train()
                for (X, y) in self.training_loader:
                    X = X.to(device=self.device)
                    y = y.to(device=self.device)
                    output = self.model(X)
                    loss = criterion(output, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.update(int(100 / epoch))

        torch.cuda.empty_cache()
        self.model.to('cpu')

    def test(self):
        print("### Client Testing...")
        self.model = self.model.to(device=self.device)
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss().cuda()
        accuracy = loss = 0

        with tqdm(total=len(self.test_loader), colour='blue') as pbar:
            for (X, y) in self.test_loader:
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                output = self.model(X)
                loss += criterion(output, y).item()
                predicted = output.argmax(dim=1, keepdim=True)
                accuracy += predicted.eq(y.view_as(predicted)).sum().item()
                pbar.update(1)

        accuracy /= len(self.test_loader.dataset)
        loss /= len(self.test_loader)
        self.accuracy = accuracy
        self.loss = loss

        torch.cuda.empty_cache()
        self.model.to('cpu')

        return accuracy, loss

    def update_model(self, m_stat):
        """Update model before a new round of training"""

        self.model.load_state_dict(m_stat)

    def get_model_state(self):
        return copy.deepcopy(self.model.state_dict())