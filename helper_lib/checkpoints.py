import torch
import os

def save_checkpoint(model, optimizer, epoch, test_loss, test_accuracy, path):

    checkpoint = {
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":test_loss,
        "accuracy":test_accuracy
    }

    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint