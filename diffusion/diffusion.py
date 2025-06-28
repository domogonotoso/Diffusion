# diffusion/diffusion.py

import torch
import torch.nn.functional as F
from tqdm import tqdm


class Diffusion:
    """
    Implements forward (q) and reverse (p) processes for DDPM.
    """

    def __init__(self, model, scheduler, device="cuda"):
        self.model = model
        self.scheduler = scheduler.to(device)
        self.device = device
        self.timesteps = scheduler.timesteps

    def q_sample(self, x_start, t, noise=None): # Add noise to x_0 to get x_t
        """
        Add noise to input image x_0 at timestep t to obtain x_t.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self.scheduler.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.scheduler.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def p_sample(self, x_t, t): # Denoise a single timestep
        """
        Sample from p(x_{t-1} | x_t) using model's noise prediction.

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * predicted_noise) + noise
        """
        betas = self.scheduler.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha = (1.0 / self.scheduler.alphas[t]).sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.scheduler.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)

        predicted_noise = self.model(x_t, t)

        mean = sqrt_recip_alpha * (x_t - betas / sqrt_one_minus_alpha_bar * predicted_noise)

        if t[0] == 0:
            return mean  # no noise at t=0
        else:
            noise = torch.randn_like(x_t)
            posterior_var = self.scheduler.posterior_variance[t].view(-1, 1, 1, 1)
            return mean + torch.sqrt(posterior_var) * noise

    def sample(self, batch_size=16, image_size=32, channels=3):  # cumulative from p_sample()
        """
        Start from random noise and iteratively denoise to get final image.
        """
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)

        return x

    def loss(self, x_0):
        """
        Training loss: predict noise from x_t, MSE between real and predicted noise.
        """
        batch_size = x_0.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)
