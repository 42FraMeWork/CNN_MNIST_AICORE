import torch
import torch.nn.functional as F

from torchvision.datasets import mnist

class ModelCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(kernel_size=4, stride=1, padding=1)

    def forward():
        whatevs


cnn = ModelCNN()

cnn(dataset)


