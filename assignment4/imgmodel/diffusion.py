import torch
import torch.nn.functional as F


class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # variance of q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, values, t, shape):
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        # forward process: x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, xt, t):
        # single reverse step: predict noise with the UNet, then sample x_{t-1}
        betas_t = self._extract(self.betas, t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, xt.shape)

        predicted_noise = model(xt, t)

        model_mean = sqrt_recip_alphas_t * (
            xt - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        )

        if (t == 0).all():
            return model_mean

        posterior_variance_t = self._extract(self.posterior_variance, t, xt.shape)
        noise = torch.randn_like(xt)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_shape, num_samples, device):
        # full reverse loop, starting from pure Gaussian noise
        model.eval()

        x = torch.randn(num_samples, *image_shape, device=device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)

        return x
