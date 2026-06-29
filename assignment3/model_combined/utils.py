import csv
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

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

def get_prediction_transform():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD
        )
    ])

    return transform

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def create_project_dirs(base_dir):
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'models'
    result_dir = base_dir / 'results'

    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    return data_dir, model_dir, result_dir

def save_training_log(history, result_dir, filename="training_log.csv"):
    if len(history) == 0:
        return

    log_path = result_dir / filename

    fieldnames = list(history[0].keys())

    with open(log_path, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
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

def plot_gan_training_curves(history, result_dir):
    epochs = [item["epoch"] for item in history]
    d_losses = [item["d_loss"] for item in history]
    g_losses = [item["g_loss"] for item in history]

    plt.figure()
    plt.plot(epochs, d_losses, label="Discriminator Loss")
    plt.plot(epochs, g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.savefig(result_dir / "gan_training_curves.png")
    plt.close()