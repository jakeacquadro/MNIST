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
            nn.Linear(512,512),
            nn.ReLU(),
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
