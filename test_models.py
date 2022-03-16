import torch

from nn_models import MultilayerPerceptron, ConvolutionalNeuralNetwork
from get_data import get_test_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataloader = get_test_dataloader()

'''
Test MLP
'''

MLP = MultilayerPerceptron(device)
MLP.load_state_dict('multilayer_perceptron.pth')
MLP.eval()