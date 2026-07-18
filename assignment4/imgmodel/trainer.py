import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from .ebm import SampleBuffer, langevin_dynamics
from .utils import (
    save_training_log,
    plot_diffusion_training_curves,
    plot_ebm_training_curves
)


def train_ebm(model, train_loader, device, model_dir, result_dir, epochs=10, learning_rate=1e-4,
        langevin_steps=60, langevin_step_size=10.0, langevin_noise=0.005, alpha=0.1, buffer_size=8192):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.0, 0.999))

    model = model.to(device)
    buffer = SampleBuffer(max_size=buffer_size)

    history = []

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_energy_real = 0.0
        running_energy_fake = 0.0

        for images, _ in train_loader:
            real_images = images.to(device)
            batch_size = real_images.size(0)
            image_shape = real_images.shape[1:]

            fake_images = buffer.sample(batch_size, image_shape, device)

            with torch.enable_grad():
                fake_images = langevin_dynamics(
                    model,
                    fake_images,
                    steps=langevin_steps,
                    step_size=langevin_step_size,
                    noise_scale=langevin_noise
                )

            energy_real = model(real_images)
            energy_fake = model(fake_images)

            cd_loss = energy_real.mean() - energy_fake.mean()
            reg_loss = alpha * (energy_real ** 2 + energy_fake ** 2).mean()
            loss = cd_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            buffer.push(fake_images)

            running_loss += loss.item()
            running_energy_real += energy_real.mean().item()
            running_energy_fake += energy_fake.mean().item()

        avg_loss = running_loss / len(train_loader)
        avg_energy_real = running_energy_real / len(train_loader)
        avg_energy_fake = running_energy_fake / len(train_loader)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "energy_real": round(avg_energy_real, 4),
            "energy_fake": round(avg_energy_fake, 4)
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"E(real): {avg_energy_real:.4f} | "
            f"E(fake): {avg_energy_fake:.4f}"
        )

        # CD loss is not a meaningful model-selection metric, so instead of tracking a
        # "best" checkpoint we save a sample grid each epoch to judge quality visually.
        grid = buffer.sample(16, image_shape, device, reinit_prob=0.0)
        grid = ((grid + 1) / 2).clamp(0, 1)
        save_image(grid, result_dir / f"ebm_samples_epoch_{epoch + 1:02d}.png", nrow=4)

        save_training_log(
            history=history,
            result_dir=result_dir,
            filename="ebm_training_log.csv"
        )

    torch.save(model.state_dict(), model_dir / "final_ebm.pth")

    # Persist the replay buffer so inference can initialize Langevin chains from it.
    torch.save(torch.stack(buffer.buffer), model_dir / "ebm_sample_buffer.pth")

    plot_ebm_training_curves(
        history=history,
        result_dir=result_dir
    )

    print("EBM training completed.")
    print(f"Final EBM model saved to {model_dir / 'final_ebm.pth'}")
    print(f"Replay buffer saved to {model_dir / 'ebm_sample_buffer.pth'}")

    return history

def train_diffusion(model, scheduler, train_loader, device, model_dir, result_dir, epochs=10, learning_rate=2e-4):
    # Trains a DDPM-style diffusion model: sample a random timestep t and noise,
    # build x_t via the forward process, and have the UNet predict that noise.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    history = []
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(images)

            noisy_images = scheduler.q_sample(images, t, noise=noise)
            predicted_noise = model(noisy_images, t)

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6)
        }

        history.append(epoch_result)

        print(f"Epoch {epoch + 1}/{epochs} | Diffusion Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save(model.state_dict(), model_dir / "best_diffusion.pth")

            print(f"Best diffusion model saved at epoch {epoch + 1}")

        save_training_log(
            history=history,
            result_dir=result_dir,
            filename="diffusion_training_log.csv"
        )

    torch.save(model.state_dict(), model_dir / "final_diffusion.pth")

    plot_diffusion_training_curves(
        history=history,
        result_dir=result_dir
    )

    print("Diffusion training completed.")
    print(f"Best diffusion model saved to {model_dir / 'best_diffusion.pth'}")
    print(f"Final diffusion model saved to {model_dir / 'final_diffusion.pth'}")

    return history
