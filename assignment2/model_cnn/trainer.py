import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .evaluator import evaluate
from .checkpoints import save_checkpoint
from .utils import (
    save_training_log,
    save_training_summary,
    plot_training_curves
)

def train_model(model, train_loader, test_loader, device, model_dir, result_dir, epochs=10, learning_rate = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        test_loss, test_accuracy = evaluate(
            model = model,
            data_loader= test_loader,
            criterion=criterion,
            device=device
        )

        epoch_result = {
            "epoch":epoch + 1,
            "train_loss":round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 2),
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_accuracy, 2)
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

            save_checkpoint(
                model = model,
                optimizer = optimizer,
                epoch = epoch + 1,
                test_loss = test_loss,
                test_accuracy = test_accuracy,
                path = model_dir / "best_model.pth"
            )

            print(f"Best model saved at epoch {epoch + 1}")

    save_checkpoint(
        model=model,
        optimizer = optimizer,
        epoch = epochs,
        test_loss = history[-1]["test_loss"], # last element
        test_accuracy=history[-1]["test_accuracy"],
        path =model_dir / "final_model.pth"
    )

    save_training_summary(
        history=history,
        best_epoch=best_epoch,
        best_accuracy=round(best_accuracy, 2),
        result_dir=result_dir
    )

    plot_training_curves(
        history=history,
        result_dir=result_dir
    )

    print("Training completed.")
    print(f"Best model saved to {model_dir / 'best_model.pth'}")
    print(f"Final model saved to {model_dir / 'final_model.pth'}")

    return history