# diffusion/scheduler.py

import torch
import numpy as np


class DiffusionScheduler:
    """
    Scheduler to precompute betas and related terms for diffusion process.
    Supports linear beta schedule.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type="linear"):
        self.timesteps = timesteps

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise NotImplementedError(f"Schedule type {schedule_type} not supported.")

        # Precompute alphas
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)  # \bar{α}_t
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]], dim=0)

        # Posterior variance used in reverse process
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def get_params(self, t):
        """
        Returns precomputed alpha/variance terms for timestep t.
        """
        return {
            "beta": self.betas[t],
            "alpha": self.alphas[t],
            "alpha_cumprod": self.alpha_cumprod[t],
            "alpha_cumprod_prev": self.alpha_cumprod_prev[t],
            "posterior_variance": self.posterior_variance[t],
        }

    def to(self, device):
        """
        Move all tensors to specified device.
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
