import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(3 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3,16,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        # app
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # 2
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        #    3
        self.conv3 = nn.Conv2d(32,64,kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 4
        self.conv4 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)


        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4*4,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        # Conf and pooling layers
        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn3(self.conv3(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn4(self.conv4(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool

        # Flatten the feature map
        x = x.view(-1, 128 * 2 * 2)

        # Fully connected layer app with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Fully connected layer 2 (output)
        x = self.fc2(x)
        return x

def get_model(model_name):
    if model_name == "MLP":
        return MLP()
    elif model_name == "SimpleCNN":
        return SimpleCNN()
    elif model_name == "EnhancedCNN":
        return EnhancedCNN()
    else:
        raise ValueError(f"Model {model_name} not recognized.")