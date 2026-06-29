import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # layer one
        self.conv1 = nn.Conv2d(3,16, kernel_size=3, stride = 1, padding=1)

        # layer two
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, stride=1,padding=1)

        # pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.flatten = nn.Flatten()

        # fully connected layer
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # layer app
        x = self.pool(F.relu(self.conv1(x)))
        # layer 2
        x = self.pool(F.relu(self.conv2(x)))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x