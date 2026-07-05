from pathlib import Path
import io

import torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from torchvision.utils import save_image

from model_combined.model import Generator

app = FastAPI(
    title = "MNIST GAN API",
    description = "API for generating MNIST images using a pre-trained GAN model.",
    version = "1.0.0"
)

# config
NOISE_DIM = 100
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_generator.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load generator
generator = Generator(noise_dim=NOISE_DIM).to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Generator model not found at {MODEL_PATH}")

generator.load_state_dict(torch.load(MODEL_PATH,map_location=device))

generator.eval()

@app.get("/")
def root():
    return {
        "message": "MNIST GAN API is running",
        "device":str(device),
        "model_path": str(MODEL_PATH)
    }

@app.get("/generate")
def generate_digit():
    with torch.no_grad():
        noise = torch.randn(1, NOISE_DIM).to(device)

        fake_image = generator(noise)

        # Generator uses Tanh, so output is in [-1, 1].
        # Convert it back to [0, 1] for image display.
        fake_image = (fake_image + 1) / 2

        image_buffer = io.BytesIO()
        save_image(fake_image, image_buffer, format="PNG")
        image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png"
    )


@app.get("/generate_batch")
def generate_batch(
    num_images: int = Query(default=16, ge=1, le=64)
):
    with torch.no_grad():
        noise = torch.randn(num_images, NOISE_DIM).to(device)

        fake_images = generator(noise)

        # Convert from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2

        image_buffer = io.BytesIO()
        save_image(
            fake_images,
            image_buffer,
            format="PNG",
            nrow=8
        )
        image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png"
    )