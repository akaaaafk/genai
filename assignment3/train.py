from pathlib import Path

from model_combined.model import Generator, Discriminator
from model_combined.data_loader import get_gan_mnist_loader
from model_combined.trainer import train_gan
from model_combined.utils import get_device, create_project_dirs

import torch
import numpy as np

torch.manual_seed(920)
np.random.seed(920)

def main():
    base_dir = Path(__file__).resolve().parent

    data_dir, model_dir, result_dir = create_project_dirs(base_dir)

    device = get_device()

    print("Using device:" ,device)

    batch_size = 64
    epochs = 10
    learning_rate = 0.0002

    train_loader= get_gan_mnist_loader(data_dir=data_dir, batch_size=batch_size)

    gen = Generator(noise_dim=100).to(device)
    dis = Discriminator().to(device)

    train_gan(
        generator=gen,
        discriminator=dis,
        train_loader=train_loader,
        device=device,
        model_dir=model_dir,
        result_dir=result_dir,
        epochs=epochs,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    main()