import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import CNN

torch.manual_seed(920)
np.random.seed(920)

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

BATCH_SIZE = 64

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(
        root = "./data",
        train=True,
        download = True,
        transform = transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root = "./data",
        train = False,
        download=True,
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle = True
    )

    test_dataset = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle = False
    )

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        accuracy = evaluate(model, test_loader, device)