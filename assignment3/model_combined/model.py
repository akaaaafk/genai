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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64,128,kernel_size=4, stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*7*7, 1)

    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class Generator(nn.Module):
    def __init__(self, noise_dim = 100):
        super(Generator, self).__init__()
        self.latent_dim = noise_dim

        self.fc1 = nn.Linear(self.latent_dim, 128*7*7)

        self.deconv1 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2, padding=1)
        self.act2 = nn.Tanh()

    def forward(self,x):
        x = self.fc1(x)
        x = x.view(x.size(0),128,7,7)
        x = self.act1(self.bn1(self.deconv1(x)))
        x = self.act2(self.deconv2(x))
        return x