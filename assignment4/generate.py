from pathlib import Path
import argparse

import torch
from torchvision.utils import save_image

from imgmodel.model import EnergyModel, UNet
from imgmodel.diffusion import DiffusionScheduler
from imgmodel.ebm import sample_ebm


IMAGE_SHAPE = (3, 64, 64)
DIFFUSION_TIMESTEPS = 1000


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10-like images locally.")
    parser.add_argument(
        "--model",
        choices=["diffusion", "ebm", "both"],
        default="diffusion",
        help="Which model to sample from (default: diffusion).",
    )
    parser.add_argument("--num", type=int, default=16, help="Number of images (1-64).")
    parser.add_argument("--steps", type=int, default=60, help="EBM Langevin steps.")
    parser.add_argument("--step-size", type=float, default=10.0, help="EBM Langevin step size.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: assignment4/results).",
    )
    return parser.parse_args()


def load_state(model, path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Loaded {path.name}")
    return model


def to_display(images):
    return ((images + 1) / 2).clamp(0, 1)


def generate_diffusion(out_path, num, device, model_dir):
    model = UNet(in_channels=IMAGE_SHAPE[0], base_channels=64, time_dim=256).to(device)
    load_state(model, model_dir / "best_diffusion.pth", device)

    scheduler = DiffusionScheduler(timesteps=DIFFUSION_TIMESTEPS, device=device)
    print(f"Sampling {num} diffusion images ({DIFFUSION_TIMESTEPS} reverse steps)...")
    images = scheduler.sample(model, IMAGE_SHAPE, num, device)

    save_image(to_display(images), out_path, nrow=min(8, num))
    print(f"Saved {out_path}")


def generate_ebm(out_path, num, steps, step_size, device, model_dir):
    model = EnergyModel(in_channels=IMAGE_SHAPE[0]).to(device)
    load_state(model, model_dir / "final_ebm.pth", device)

    buffer_path = model_dir / "ebm_sample_buffer.pth"
    init = None
    if buffer_path.exists():
        buffer = torch.load(buffer_path, map_location="cpu")
        indices = torch.randint(buffer.size(0), (num,))
        init = buffer[indices].clone()
        print(f"Initialized Langevin from {buffer_path.name} ({buffer.size(0)} samples)")
    else:
        print(
            f"WARNING: {buffer_path} not found. Starting from noise — "
            "raise --steps substantially or retrain to regenerate the buffer."
        )

    print(f"Sampling {num} EBM images ({steps} Langevin steps)...")
    images = sample_ebm(
        model,
        image_shape=IMAGE_SHAPE,
        num_samples=num,
        device=device,
        steps=steps,
        step_size=step_size,
        init=init,
    )

    save_image(to_display(images), out_path, nrow=min(8, num))
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    if not 1 <= args.num <= 64:
        raise SystemExit("--num must be between 1 and 64")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    out_dir = args.out_dir or (base_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.model in ("diffusion", "both"):
        generate_diffusion(
            out_dir / "generated_diffusion.png",
            args.num,
            device,
            model_dir,
        )

    if args.model in ("ebm", "both"):
        generate_ebm(
            out_dir / "generated_ebm.png",
            args.num,
            args.steps,
            args.step_size,
            device,
            model_dir,
        )


if __name__ == "__main__":
    main()
