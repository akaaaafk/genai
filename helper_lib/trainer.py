import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .evaluator import evaluate, evaluate_rnn
from .checkpoints import save_checkpoint
from .utils import (
    save_training_log,
    save_training_summary,
    plot_gan_training_curves,
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

def train_rnn(model, train_loader, test_loader, device, model_dir, result_dir, epochs=10,learning_rate=0.001,max_grad_norm=5):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    model = model.to(device)

    model_dir.mkdir(parents = True, exist_ok = True)
    result_dir.mkdir(parents=True, exist_ok=True)

    history = []

    best_test_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        total_tokens = 0

        for inputs, targets in train_loader:
            inputs = inputs.long().to(device)
            targets = targets.long().to(device)

            # inputs: (batch_size, seq_len)
            # targets: (batch_size, seq_len)

            outputs, hidden = model(inputs)

            vocab_size = outputs.size(-1)

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(
                model.parameters(),max_norm=max_grad_norm
            )
            optimizer.step()

            num_tokens = targets.numel()

            running_loss +=loss.item() * num_tokens
            total_tokens +=num_tokens

        train_loss = running_loss / total_tokens
        train_perplexity = float(np.exp(min(train_loss,20)))

        test_loss, test_perplexity = evaluate_rnn(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_perplexity": round(train_perplexity, 4),
            "test_loss": round(test_loss, 4),
            "test_perplexity": round(test_perplexity, 4)
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train PPL: {train_perplexity:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test PPL: {test_perplexity:.4f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_loss,
                    "test_perplexity": test_perplexity
                },
                model_dir / "best_rnn.pth"
            )

            print(f"Best RNN model saved at epoch {epoch + 1}")
        save_training_log(
            history=history,
            result_dir=result_dir
        )

        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": history[-1]["test_loss"],
                "test_perplexity": history[-1]["test_perplexity"]
            },
            model_dir / "final_rnn.pth"
        )

        print("RNN training completed.")
        print(f"Best RNN model saved to {model_dir / 'best_rnn.pth'}")
        print(f"Final RNN model saved to {model_dir / 'final_rnn.pth'}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best test loss: {best_test_loss:.4f}")

        return history

def train_gan(generator, discriminator, train_loader, device, model_dir, result_dir,epochs=10,noise_dim=100, learning_rate = 0.0002):
    criterion = nn.BCEWithLogitsLoss()

    optim_g = optim.Adam(
        generator.parameters(),
        lr = learning_rate,
        betas = (0.5,0.999)
    )

    optim_d = optim.Adam(
        discriminator.parameters(),
        lr = learning_rate,
        betas = (0.5,0.999)
    )

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    history = []

    best_g_loss = float("inf")
    best_epoch = 0

    best_d_loss = float("inf")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        running_d_loss = 0.0
        running_g_loss = 0.0

        for images, _ in train_loader:
            real_images = images.to(device)
            batch_size = images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # train discriminator
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(noise)

            real_outputs = discriminator(real_images)
            fake_outputs = discriminator(fake_images.detach())

            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_fake = criterion(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()

            # train generator
            noise = torch.randn(batch_size,noise_dim).to(device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images)

            g_loss = criterion(fake_outputs, real_labels)

            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        avg_d_loss = running_d_loss / len(train_loader)
        avg_g_loss = running_g_loss / len(train_loader)

        epoch_result = {
            "epoch": epoch + 1,
            "d_loss": round(avg_d_loss, 4),
            "g_loss": round(avg_g_loss, 4)
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"D Loss: {avg_d_loss:.4f} | "
            f"G Loss: {avg_g_loss:.4f}"
        )

        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_epoch = epoch + 1

            torch.save(
                generator.state_dict(),
                model_dir / "best_generator.pth"
            )

        if avg_d_loss < best_d_loss:
            best_d_loss = avg_d_loss

            torch.save(
                discriminator.state_dict(),
                model_dir / "best_discriminator.pth"
            )

        save_training_log(
            history=history,
            result_dir=result_dir
        )

    torch.save(
        generator.state_dict(),
        model_dir / "generator.pth"
    )
    torch.save(
        discriminator.state_dict(),
        model_dir / "discriminator.pth"
    )

    plot_gan_training_curves(
        history=history,
        result_dir=result_dir
    )

    print("GAN training completed.")
    print(f"Generator saved to {model_dir / 'generator.pth'}")
    print(f"Discriminator saved to {model_dir / 'discriminator.pth'}")

    return history
