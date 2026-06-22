import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Normalize dataset to get a faster speed of gradient descent

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = CIFAR10_MEAN,
        std = CIFAR10_STD
    )
])

def get_data_loaders(data_dir, batch_size = 64):
    train_dataset = datasets.CIFAR10(
        root = data_dir,
        train=True,
        download = True,
        transform = transform
    )

    test_dataset = datasets.CIFAR10(
        root = data_dir,
        train = False,
        download = True,
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False
    )

    return train_loader, test_loader