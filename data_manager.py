import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class Sharding(Dataset):
    """A sharding of a dataset"""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DatasetCreator(object):
    """Dataset creator loads and divides dataset"""

    def __init__(self, num_clients, dataset_name, partition_method, alpha=0.5, seed=5):
        random.seed(seed)  # fix seed for reproduction
        self.n_clients = num_clients
        self.dt_name = dataset_name
        self.pt_method = partition_method
        self.shardings = []
        self.training_data, self.test_data = self.__get_dataset()
        self.__divide_dataset(alpha=alpha)

    def get_loader(self, client_id=1, batch_sz=20, is_test=False, pin_memory=True, time_out=0, num_workers=0):
        """Get DataLoader object"""
        if client_id < 1 or client_id > self.n_clients:
            print("ERROR: Client ID out of range. Should be in [1, num_clients].")
            exit(-1)

        if not is_test:
            sharding = Sharding(self.training_data, self.shardings[client_id-1])
            loader = DataLoader(sharding, batch_sz, shuffle=True, pin_memory=pin_memory, timeout=time_out,
                                num_workers=num_workers, drop_last=True)
        else:
            index = [_ for _ in range(len(self.test_data))]
            non_sharding = Sharding(self.test_data, index)
            loader = DataLoader(non_sharding, batch_sz, shuffle=False, pin_memory=pin_memory, timeout=time_out,
                                num_workers=num_workers, drop_last=False)
        return loader

    def get_training_data_len(self, client_id=-1):
        if client_id == -1:
            return sum(map(len, self.shardings))
        return len(self.shardings[client_id-1])

    def get_test_data_len(self):
        return len(self.test_data)

    def __divide_dataset(self, alpha=0.5):
        if self.pt_method == 'uniform':
            data_len = len(self.training_data)
            indexes = list(range(data_len))
            random.shuffle(indexes)

            for _ in range(self.n_clients):
                part_len = int(1. / self.n_clients * data_len)
                self.shardings.append(indexes[0:part_len])
                indexes = indexes[part_len:]

        elif self.pt_method == 'dirichlet':
            labels = torch.as_tensor(self.training_data.targets)
            num_classes = labels.max() + 1
            label_distribution = np.random.dirichlet([alpha] * self.n_clients, num_classes)
            class_idx = [np.argwhere(labels == y).flatten() for y in range(num_classes)]
            client_idx = [[] for _ in range(self.n_clients)]

            for c, fracs in zip(class_idx, label_distribution):
                for i, idx in enumerate(np.split(c, ((np.cumsum(fracs)[:-1]) * len(c)).astype(int))):
                    client_idx[i] += [idx]

            self.shardings = [np.concatenate(idx) for idx in client_idx]

        else:
            print("ERROR: Unknown partition method.")
            exit(-1)

    def __get_dataset(self):
        training_dataset = test_dataset = None
        training_transform, test_transform = self.__get_transform()

        if self.dt_name == 'MNIST':
            training_dataset = datasets.MNIST(
                root='data/MNIST',
                train=True,
                download=True,
                transform=training_transform
            )
            test_dataset = datasets.MNIST(
                root='data/MNIST',
                train=False,
                download=True,
                transform=test_transform
            )

        else:
            print("ERROR: Unknown dataset type.")
            exit(-1)

        return training_dataset, test_dataset

    def __get_transform(self):
        training_transform = test_transform = None

        if self.dt_name == 'MNIST':
            training_transform = transforms.ToTensor()
            test_transform = transforms.ToTensor()

        else:
            print("ERROR: Unknown dataset type.")
            exit(-1)

        return training_transform, test_transform