from typing import Optional, Tuple

import torch

from .base import NoiseScheduler


class LinearNoiseScheduler(NoiseScheduler):
    """
    Linear noise scheduler implementation for DDPM.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the linear noise scheduler.

        Args:
            num_timesteps (int): Number of diffusion timesteps.
            beta_start (float): Starting beta value (noise level).
            beta_end (float): Ending beta value (noise level).
            device (torch.device, optional): Device for tensors.
        """
        super().__init__(num_timesteps, device)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.set_timesteps()

    def set_timesteps(self):
        """
        Set up the noise schedule parameters using a linear beta schedule.
        Computes all required parameters for the diffusion process.
        """
        # Initialize beta schedule
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_timesteps, device=self.device
        )

        # Compute alpha parameters
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
