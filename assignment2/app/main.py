import io
from pathlib import Path

import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File
from PIL import Image

from model_cnn.model import CNN
from model_cnn.checkpoints import load_checkpoint
from model_cnn.utils import CLASSES, get_device, get_prediction_transform


app = FastAPI(title="Assignment 2 CIFAR10 CNN API")


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"


device = get_device()

model = CNN(num_classes=10).to(device)

load_checkpoint(
    model=model,
    checkpoint_path=MODEL_PATH,
    device=device
)

model.eval()

transform = get_prediction_transform()


@app.get("/")
def root():
    return {
        "message": "Assignment 2 CIFAR10 CNN API is running",
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_index = predicted.item()
    predicted_label = CLASSES[predicted_index]

    return {
        "predicted_class": predicted_label,
        "class_index": predicted_index,
        "confidence": round(confidence.item(), 4)
    }