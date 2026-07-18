from pathlib import Path
import io

import torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from torchvision.utils import save_image

from imgmodel.model import EnergyModel, UNet
from imgmodel.diffusion import DiffusionScheduler
from imgmodel.ebm import sample_ebm

app = FastAPI(
    title="CIFAR-10 Generative Models API",
    description="API for generating CIFAR-10-like images using an Energy-Based Model and a Diffusion Model.",
    version="1.0.0"
)

# config
IMAGE_SHAPE = (3, 64, 64)
DIFFUSION_TIMESTEPS = 1000

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

# Serve the final EBM checkpoint: CD training loss is not a quality metric, so the
# "best by loss" checkpoint was effectively the epoch-1 model.
EBM_PATH = MODEL_DIR / "final_ebm.pth"
EBM_BUFFER_PATH = MODEL_DIR / "ebm_sample_buffer.pth"
DIFFUSION_PATH = MODEL_DIR / "best_diffusion.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
energy_model = EnergyModel(in_channels=IMAGE_SHAPE[0]).to(device)
diffusion_model = UNet(in_channels=IMAGE_SHAPE[0], base_channels=64, time_dim=256).to(device)
scheduler = DiffusionScheduler(timesteps=DIFFUSION_TIMESTEPS, device=device)

def _load_weights(model, path, name):
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {name} weights from {path}")
    else:
        print(f"WARNING: {name} checkpoint not found at {path}. Serving randomly initialized weights.")

    model.eval()
    return model

energy_model = _load_weights(energy_model, EBM_PATH, "EBM")
diffusion_model = _load_weights(diffusion_model, DIFFUSION_PATH, "Diffusion UNet")

# The EBM was trained with a persistent replay buffer, so short Langevin chains only
# work when initialized from buffer samples. Pure-noise starts produce noise.
if EBM_BUFFER_PATH.exists():
    ebm_buffer = torch.load(EBM_BUFFER_PATH, map_location="cpu")
    print(f"Loaded EBM replay buffer with {ebm_buffer.size(0)} samples from {EBM_BUFFER_PATH}")
else:
    ebm_buffer = None
    print(f"WARNING: EBM replay buffer not found at {EBM_BUFFER_PATH}. "
          "Langevin chains will start from noise, which needs far more steps.")

def _ebm_init(num_samples):
    if ebm_buffer is None:
        return None

    indices = torch.randint(ebm_buffer.size(0), (num_samples,))
    return ebm_buffer[indices].clone()

def _images_to_png(images, nrow=8):
    # both models are trained on images normalized to [-1, 1]; convert back to [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    image_buffer = io.BytesIO()
    save_image(images, image_buffer, format="PNG", nrow=nrow)
    image_buffer.seek(0)

    return image_buffer

@app.get("/")
def root():
    return {
        "message": "CIFAR-10 Generative Models API is running",
        "device": str(device),
        "ebm_checkpoint": str(EBM_PATH),
        "diffusion_checkpoint": str(DIFFUSION_PATH)
    }

@app.get("/generate/ebm")
def generate_ebm_image(
    steps: int = Query(default=60, ge=1, le=500),
    step_size: float = Query(default=10.0, gt=0)
):
    image = sample_ebm(
        energy_model,
        image_shape=IMAGE_SHAPE,
        num_samples=1,
        device=device,
        steps=steps,
        step_size=step_size,
        init=_ebm_init(1)
    )

    return StreamingResponse(
        _images_to_png(image, nrow=1),
        media_type="image/png"
    )

@app.get("/generate/ebm/batch")
def generate_ebm_batch(
    num_images: int = Query(default=16, ge=1, le=64),
    steps: int = Query(default=60, ge=1, le=500),
    step_size: float = Query(default=10.0, gt=0)
):
    images = sample_ebm(
        energy_model,
        image_shape=IMAGE_SHAPE,
        num_samples=num_images,
        device=device,
        steps=steps,
        step_size=step_size,
        init=_ebm_init(num_images)
    )

    return StreamingResponse(
        _images_to_png(images, nrow=8),
        media_type="image/png"
    )

@app.get("/generate/diffusion")
def generate_diffusion_image():
    image = scheduler.sample(
        diffusion_model,
        image_shape=IMAGE_SHAPE,
        num_samples=1,
        device=device
    )

    return StreamingResponse(
        _images_to_png(image, nrow=1),
        media_type="image/png"
    )

@app.get("/generate/diffusion/batch")
def generate_diffusion_batch(
    num_images: int = Query(default=16, ge=1, le=64)
):
    images = scheduler.sample(
        diffusion_model,
        image_shape=IMAGE_SHAPE,
        num_samples=num_images,
        device=device
    )

    return StreamingResponse(
        _images_to_png(images, nrow=8),
        media_type="image/png"
    )
