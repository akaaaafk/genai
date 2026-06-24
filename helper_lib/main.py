from pathlib import Path

import torch
import numpy as np
from model import get_model

import model
from data_loader import get_data_loaders
from trainer import train_model
from utils import get_best_device,get_available_devices, create_project_dirs

torch.manual_seed(920)

def main():
    base_dir = Path(__file__).resolve().parent

    data_dir, model_dir, result_dir = create_project_dirs(base_dir)

    devices = get_available_devices()
    print("Available devices:", devices)
    device = get_best_device()
    print("Using device:", device)

    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    train_loader, test_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size)

    model = get_model("SimpleCNN").to(device)

    train_model(
        model = model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model_dir=model_dir,
        result_dir=result_dir,
        epochs=epochs,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    main()