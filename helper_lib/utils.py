import csv
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2470, 0.2435, 0.2616),
    },
    "mnist": {
        "mean": (0.1307,),
        "std":  (0.3081,),
    },
    "fashion_mnist": {
        "mean": (0.2860,),
        "std":  (0.3530,),
    },
}

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

def get_prediction_transform(dataset_name = "cifar10", image_size = (64, 64)):
    stats = DATASET_STATS[dataset_name]

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = stats["mean"],
            std = stats["std"]
        )
    ])

    return transform

def get_available_devices():
    devices = []

    # NVIDIA GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))

    # Intel GPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            devices.append(torch.device(f"xpu:{i}"))

    # Apple Silicon GPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    # CPU 永远保底
    devices.append(torch.device("cpu"))

    return devices

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def create_project_dirs(base_dir):
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'models'
    result_dir = base_dir / 'results'

    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    return data_dir, model_dir, result_dir

def save_training_log(history, result_dir):
    log_path = result_dir / "training_log.csv"

    with open(log_path, mode = "w", newline = "") as file:
        writer = csv.DictWriter(
            file,
            fieldnames = [
                "epoch",
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy"
            ]
        )

        writer.writeheader()

        for row in history:
            writer.writerow(row)

    print(f"Training log saved to {log_path}")

def save_training_summary(history, best_epoch, best_accuracy, result_dir):
    summary = {
        "total_epochs": len(history),
        "best_epoch": best_epoch,
        "best_test_accuracy": best_accuracy,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_loss": history[-1]["test_loss"],
        "final_test_accuracy": history[-1]["test_accuracy"]
    }

    summary_path = result_dir / "training_summary.json"

    with open(summary_path, mode="w") as file:
        json.dump(summary, file, indent=4)

    print(f"Training summary saved to {summary_path}")

def plot_training_curves(history, result_dir):
    epochs = [row["epoch"] for row in history]

    train_losses = [row["train_loss"] for row in history]
    test_losses = [row["test_loss"] for row in history]

    train_accuracies = [row["train_accuracy"] for row in history]
    test_accuracies = [row["test_accuracy"] for row in history]

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.savefig(result_dir / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.savefig(result_dir / "accuracy_curve.png")
    plt.close()

    print(f"Loss curve saved to {result_dir / 'loss_curve.png'}")
    print(f"Accuracy curve saved to {result_dir / 'accuracy_curve.png'}")