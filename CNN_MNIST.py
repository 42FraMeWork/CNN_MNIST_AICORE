import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from statistics import mean

train_data = datasets.MNIST(root='MNIST-data',                       
                            transform=transforms.ToTensor(),        
                            train=True,                               
                            download=True                            
                           )

test_data = datasets.MNIST(root='MNIST-data',
                           transform=transforms.ToTensor(),
                           train=False
                          )

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)

test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=32)

class ModelCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=50176, out_features=10),
            torch.nn.Softmax()
            )

    def forward(self, x):
        return self.nn(x)


cnn = ModelCNN()

def train(model, train_loader, epochs):
    optimiser = torch.optim.Adam(model.parameters())
    losses = []
    writer = SummaryWriter()
    batch_idx = 0
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            data, labels = batch
            pred = model(data)
            loss = F.cross_entropy(pred, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            losses.append(loss)
            writer.add_scalar('Loss:', loss.item(), batch_idx)
            batch_idx += 1
    # plt.plot(losses)
    # plt.show()        
cnn = torch.load('./cnn.pth')            
# train(cnn, train_loader, 1)
# torch.save(cnn, './cnn.pth')

def test(model, test_loader):
    accuracies = []
    losses_test = []
    writer = SummaryWriter()
    batch_idx_test = 0
    for batch in tqdm(test_loader):
        data, labels = batch
        pred = model(data) 
        loss_test = F.cross_entropy(pred, labels)
        correct = labels == torch.argmax(pred, dim=1)
        accuracies.extend(correct)
        losses_test.append(loss_test)
        writer.add_scalar('Loss of test data:', loss_test.item(), batch_idx_test)
        batch_idx_test += 1
    return mean(accuracies)

accuracy = test(cnn, test_loader)
print('Acurracy:', accuracy)