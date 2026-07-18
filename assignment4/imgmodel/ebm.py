import random
import torch

class SampleBuffer:
    def __init__(self, max_size=8192):
        self.max_size = max_size
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size, image_shape, device, reinit_prob=0.05):
        if len(self.buffer) == 0:
            return torch.rand(batch_size, *image_shape, device=device) * 2 - 1

        samples = []
        for _ in range(batch_size):
            if random.random() > reinit_prob:
                samples.append(random.choice(self.buffer))
            else:
                samples.append(torch.rand(*image_shape) * 2 - 1)

        return torch.stack(samples).to(device)

    def push(self, samples):
        samples = samples.detach().cpu()

        for sample in samples:
            self.buffer.append(sample)

        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]


def langevin_dynamics(model, x, steps=60, step_size=10.0, noise_scale=0.005):
    #   x_{k+1} = x_k - step_size * dE/dx + noise
    x = x.clone().detach()

    for _ in range(steps):
        x = x + torch.randn_like(x) * noise_scale
        x.requires_grad_(True)

        energy = model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        x = x.detach() - step_size * grad
        x = x.clamp(-1, 1)

    return x.detach()

def sample_ebm(model, image_shape, num_samples, device, steps=60, step_size=10.0, noise_scale=0.005, init=None):
    # `init` lets inference start from replay-buffer samples (matching how the model
    # was trained) instead of pure noise, which short Langevin chains can't refine.
    model.eval()

    if init is None:
        x = torch.rand(num_samples, *image_shape, device=device) * 2 - 1
    else:
        x = init.to(device)

    with torch.enable_grad():
        x = langevin_dynamics(model, x, steps=steps, step_size=step_size, noise_scale=noise_scale)

    return x
