import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET_CONFIGS = {
    "cifar10": {
        "dataset": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
        "in_channels": 3,
    },
    "cifar100": {
        "dataset": datasets.CIFAR100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "in_channels": 3,
    },
    "mnist": {
        "dataset": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "num_classes": 10,
        "in_channels": 1,
    },
    "fashionmnist": {
        "dataset": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "num_classes": 10,
        "in_channels": 1,
    },
}

def get_transforms(dataset_name, image_size = 64):
    config = DATASET_CONFIGS[dataset_name]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = config["mean"],
            std = config["std"]
        )
    ])

    return transform


def get_data_loaders(dataset_name, data_dir, batch_size = 64, image_size = 64, download = True,num_workers = 0):
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Choose from {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    dataset_class = config["dataset"]

    transform = get_transforms(dataset_name, image_size=image_size)

    train_dataset = dataset_class(
        root = data_dir,
        train = True,
        download = download,
        transform = transform
    )


    test_dataset = dataset_class(
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

    return (
        train_loader,
        test_loader,
        config["num_classes"],
        config["in_channels"]
    )