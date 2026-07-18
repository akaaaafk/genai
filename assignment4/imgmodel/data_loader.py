from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)

def get_cifar10_transform(image_size=64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

def get_cifar10_loaders(data_dir, batch_size=64, image_size=64, download=True, num_workers=0):
    transform = get_cifar10_transform(image_size=image_size)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
