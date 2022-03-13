from torch import optim
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch

from evaluate_models import test_model
from get_data import get_training_dataloader, get_test_dataloader
from nn_models import MultilayerPerceptron
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_dataloader = get_training_dataloader()
test_dataloader = get_test_dataloader()

n_epochs = 20

def train_model(model, training_data, optimizer, epoch):
    model.train()

    n_steps = len(training_data)

    for index, (images, labels) in enumerate(training_data):
        img_batch = images.to(device)
        label_batch = labels.to(device)

        output = model(img_batch)
        loss = model.loss_function(output, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, n_epochs, index + 1, n_steps, loss.item()))
            # classes = torch.argmax(output, dim=1)
            # labels = labels.to(device)
            # print('Batch accuracy: %s' % (str(100*torch.mean((classes == labels).float()).item()) + '%'))

'''
Train and test multilayer perceptron
'''

MLP = MultilayerPerceptron(device)
# training_dataloader.to(device)
# optimizer = optim.Adam(MLP.parameters(), lr=0.1)
optimizer = optim.SGD(MLP.parameters(), lr=0.1)
# SGD works much better for this MLP
# learning rate of 0.1 seems to yield consistently highest accuracy on training data, however, learning rate of 0.001
# may yield the highest accuracy if the number if iterations is sufficiently high (at least >20)

for epoch in range(n_epochs):
    print('Epoch: %s' % str(epoch + 1))
    train_model(epoch=epoch, model=MLP, training_data=training_dataloader, optimizer=optimizer)
    test_model(dataloader=test_dataloader, model=MLP)

torch.save(MLP, 'multilayer_perceptron.pth')