import torch.nn as nn


class TwoNN(nn.Module):
    def __init__(self, in_features=784, num_hiddens=200, num_classes=10):
        super(TwoNN, self).__init__()
        self.activation = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=in_features, out_features=num_hiddens, bias=True)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=True)
        self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(CNN, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1,
                               stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5),
                               padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN2(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(CNN2, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1,
                               stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5),
                               padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x