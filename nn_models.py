import torch
from torch import nn, optim


class MultilayerPerceptron(nn.Module):

    def __init__(self, device):
        super(MultilayerPerceptron, self).__init__()
        self.device = device

        self.flatten = nn.Flatten()  # turns image into vector
        self.MLP_sequential = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            # nn.Sigmoid(), # sigmoid does not perform as well as ReLU in this case, but the difference is small
            nn.Linear(512,512),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(512,10) # output to 10 classes
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.MLP_sequential(x)
        return logits

class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, device):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


