from tqdm import tqdm
import torch
import torchvision
from utils.models import *


class ModelManager(object):
    """Manage one model to do training or test"""

    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('gpu')
        self.model = self.__init_model()
        self.accuracy = 0
        self.loss = 0

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

    def train(self, train_loader, epoch, lr=1e-2, momentum=0.9, weight_decay=4e-5):
        print("### Training...")
        self.model = self.model.to(device=self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss().cuda()

        step = 0
        with tqdm(total=100, colour='green') as pbar:
            while step < epoch:
                step += 1
                self.model.train()
                for (X, y) in train_loader:
                    X = X.to(device=self.device)
                    y = y.to(device=self.device)
                    output = self.model(X)
                    loss = criterion(output, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.update(int(100/epoch))

        torch.cuda.empty_cache()
        self.model.to('cpu')

    def test(self, test_loader):
        print("### Testing...")
        self.model = self.model.to(device=self.device)
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss().cuda()
        accuracy = loss = 0

        with tqdm(total=len(test_loader), colour='blue') as pbar:
            for (X, y) in test_loader:
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                output = self.model(X)
                loss += criterion(output, y).item()
                predicted = output.argmax(dim=1, keepdim=True)
                accuracy += predicted.eq(y.view_as(predicted)).sum().item()
                pbar.update(1)

        accuracy /= len(test_loader.dataset)
        loss /= len(test_loader)
        self.accuracy = accuracy
        self.loss = loss
        print("Model {} test accuracy {}%, loss {}".format(self.model_name, round(self.accuracy*100, 2),
                                                          round(self.loss, 2)))

        torch.cuda.empty_cache()
        self.model.to('cpu')

        return accuracy, loss

