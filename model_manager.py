from tqdm import tqdm
import gc
import copy
import torch
import torchvision
from utils.models import *
from utils.algrithms import FedAvg


class Aggregator(object):
    """A server"""

    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda')
        self.model = self.__init_model()
        self.accuracy = 0
        self.loss = 0
        self.collected_updates = None
        self.absolute_weights = None
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

    def init_data(self, data_creator, batch_sz, time_out=0, num_workers=0):
        self.test_loader = data_creator.get_loader(batch_sz=batch_sz, is_test=True, time_out=time_out,
                                                   num_workers=num_workers)

    def get_test_data_len(self):
        if self.test_loader is None:
            print("ERROR: not init_data yet.")
            exit(-1)
        return len(self.test_loader.dataset)

    def test(self):
        print("### Testing...")
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
        print("Model {} test accuracy {}%, loss {}".format(self.model_name, round(self.accuracy*100, 2),
                                                          round(self.loss, 2)))

        torch.cuda.empty_cache()
        self.model.to('cpu')

        return accuracy, loss

    def retrieve_update(self, client_id, update, weight):
        """Retrieve udpate from a client"""

        if self.collected_updates is None:
            self.collected_updates = {}
            self.absolute_weights = {}

        self.collected_updates[client_id] = update
        self.absolute_weights[client_id] = weight

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