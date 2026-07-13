import torch
import numpy as np

def evaluate(model, data_loader, criterion, device = 'cpu'):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def evaluate_rnn(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.long().to(device)
            targets = targets.long().to(device)

            outputs, hidden = model(inputs)

            vocab_size = outputs.size(-1)

            # squeeze first two dims
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            # calculating all elements
            num_tokens = targets.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = float(np.exp(min(avg_loss, 20)))

    return avg_loss, perplexity