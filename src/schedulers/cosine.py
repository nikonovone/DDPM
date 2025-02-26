import math
from typing import Optional, Tuple

import torch

from .base import NoiseScheduler


class CosineNoiseScheduler(NoiseScheduler):
    """
    Cosine noise scheduler implementation for DDPM.
    Based on the paper "Improved Denoising Diffusion Probabilistic Models".
    """

    def __init__(
        self,
        num_timesteps: int,
        s: float = 0.008,  # Параметр для предотвращения слишком малых значений beta
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the cosine noise scheduler.

        Args:
            num_timesteps (int): Number of diffusion timesteps.
            s (float): Small offset to prevent beta from being too small near t=0.
            device (torch.device, optional): Device for tensors.
        """
        super().__init__(num_timesteps, device)
        self.s = s
        self.set_timesteps()

    def set_timesteps(self):
        """
        Set up the noise schedule parameters using a cosine beta schedule.
        Computes all required parameters for the diffusion process.
        """
        # Compute alpha_cum_prod directly using cosine schedule
        timesteps = torch.arange(self.num_timesteps + 1, device=self.device)
        f_t = (
            torch.cos(
                ((timesteps / self.num_timesteps + self.s) / (1 + self.s))
                * (math.pi / 2)
            )
            ** 2
        )

        # Normalize f_t to get alpha_cum_prod
        f_0 = f_t[0]
        self.alpha_cum_prod = f_t / f_0

        # Compute betas and alphas from alpha_cum_prod
        self.alpha_cum_prod = self.alpha_cum_prod[:-1]  # Remove the last element
        self.betas = 1 - (self.alpha_cum_prod[1:] / self.alpha_cum_prod[:-1])
        self.betas = torch.cat([torch.tensor([0.0], device=self.device), self.betas])
        self.betas = torch.clamp(self.betas, 0.0, 0.999)

        # Recompute alpha_cum_prod for consistency
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

        # Compute sqrt terms for convenience
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)

    def sample_prev_timestep(
        self, xt: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_{t-1} given x_t and predicted noise using the reverse diffusion process.

        Args:
            xt (torch.Tensor): Current noisy sample at timestep t.
            noise_pred (torch.Tensor): Model's noise prediction.
            t (torch.Tensor): Current timestep indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Previous timestep sample x_{t-1}, Predicted clean image x_0)
        """
        # Ensure all tensors are on the same device
        device = xt.device
        t = t.to(device)

        # Compute x0 prediction
        x0 = (
            xt
            - self.sqrt_one_minus_alpha_cum_prod[t].unsqueeze(-1).unsqueeze(-1)
            * noise_pred
        ) / self.sqrt_alpha_cum_prod[t].unsqueeze(-1).unsqueeze(-1)
        x0 = torch.clamp(x0, -1.0, 1.0)

        # Compute mean for the reverse process
        beta_t = self.betas[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_cum_prod_t = (
            self.sqrt_one_minus_alpha_cum_prod[t].unsqueeze(-1).unsqueeze(-1)
        )
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).unsqueeze(-1).unsqueeze(-1)

        mean = xt - (beta_t * noise_pred) / sqrt_one_minus_alpha_cum_prod_t
        mean = mean / sqrt_alpha_t

        if (t == 0).all():
            return mean, x0

        # Compute variance and sample
        variance = (1.0 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
        variance = variance * self.betas[t]
        sigma = torch.sqrt(variance).unsqueeze(-1).unsqueeze(-1)

        # Sample from Gaussian distribution
        z = torch.randn_like(xt, device=device)
        return mean + sigma * z, x0
