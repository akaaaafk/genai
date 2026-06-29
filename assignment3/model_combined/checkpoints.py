import torch


def save_checkpoint(model, optimizer, epoch, test_loss, test_accuracy, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint