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

n_epochs = 10
# 10 epochs seems to be enough for Adam with an initial learning rate of 0.001. For SGD with the same learning rate,
# more epochs are needed to converge >20

MLP = MultilayerPerceptron(device).to(device) # must set device to GPU (cuda)
# training_dataloader.to(device)
optimizer = optim.Adam(MLP.parameters(), lr=0.001)
# optimizer = optim.SGD(MLP.parameters(), lr=0.1)
# Adam's adaptive learning rate seems to perform consistently slightly better than SGD in this scenario. However, the
# performance of SGD is still very good.

for epoch in range(n_epochs):
    print('Epoch: %s' % str(epoch + 1))
    train_model(epoch=epoch, model=MLP, training_data=training_dataloader, optimizer=optimizer)
    test_model(dataloader=test_dataloader, model=MLP)

torch.save(MLP, 'multilayer_perceptron.pth')