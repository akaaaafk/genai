import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt


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

def plot_diffusion_training_curves(history, result_dir):
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss (MSE on noise)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Diffusion Model Training Loss")
    plt.legend()
    plt.savefig(result_dir / "diffusion_training_curves.png")
    plt.close()

    print(f"Diffusion training curve saved to {result_dir / 'diffusion_training_curves.png'}")

def plot_ebm_training_curves(history, result_dir):
    epochs = [row["epoch"] for row in history]
    energy_real = [row["energy_real"] for row in history]
    energy_fake = [row["energy_fake"] for row in history]

    plt.figure()
    plt.plot(epochs, energy_real, label="E(real)")
    plt.plot(epochs, energy_fake, label="E(fake)")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title("EBM Training Energies")
    plt.legend()
    plt.savefig(result_dir / "ebm_training_curves.png")
    plt.close()

    print(f"EBM training curve saved to {result_dir / 'ebm_training_curves.png'}")
