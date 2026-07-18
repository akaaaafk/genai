from pathlib import Path

import torch
import numpy as np

from imgmodel.model import get_model
from imgmodel.data_loader import get_cifar10_loaders
from imgmodel.trainer import train_ebm
from imgmodel.utils import get_device, create_project_dirs

torch.manual_seed(920)
np.random.seed(920)

def main():
    base_dir = Path(__file__).resolve().parent

    data_dir, model_dir, result_dir = create_project_dirs(base_dir)

    device = get_device()
    print("Using device:", device)

    batch_size = 64
    epochs = 30
    learning_rate = 1e-4

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=64
    )

    model = get_model("EBM").to(device)

    train_ebm(
        model=model,
        train_loader=train_loader,
        device=device,
        model_dir=model_dir,
        result_dir=result_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        langevin_steps=60,
        langevin_step_size=10.0,
        langevin_noise=0.005,
        alpha=0.1
    )

if __name__ == "__main__":
    main()
