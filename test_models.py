import torch

from nn_models import MultilayerPerceptron, ConvolutionalNeuralNetwork
from get_data import get_test_dataloader
from validate_models import validate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataloader = get_test_dataloader()

'''
Test MLP
'''

MLP = MultilayerPerceptron(device)
MLP.load_state_dict(torch.load('multilayer_perceptron.pth'))
MLP.to(device)
MLP.eval()

validate_model(test_dataloader, MLP)


'''
Test CNN
'''

CNN = ConvolutionalNeuralNetwork(device)
CNN.load_state_dict(torch.load('convolutional_neural_network.pth'))
CNN.to(device)
CNN.eval()

validate_model(test_dataloader, CNN)