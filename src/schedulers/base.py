from abc import ABC, abstractmethod

import torch


class NoiseScheduler(ABC):
    """
    Base class for noise schedulers used in diffusion models.

    Attributes:
        num_timesteps (int): Number of diffusion timesteps.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(self, num_timesteps: int, device: torch.device = None):
        """
        Initialize the noise scheduler.

        Args:
            num_timesteps (int): Number of diffusion timesteps.
            device (torch.device, optional): Device to use (defaults to None, set later).
        """
        self.num_timesteps = num_timesteps
        self.device = device if device is not None else torch.device("cpu")

        # Placeholder for scheduler-specific parameters
        self.betas = None
        self.alphas = None
        self.alpha_cum_prod = None
        self.sqrt_alpha_cum_prod = None
        self.sqrt_one_minus_alpha_cum_prod = None

    @abstractmethod
    def set_timesteps(self):
        """
        Abstract method to set up the noise schedule (betas, alphas, etc.).
        Must be implemented by subclasses.
        """
        pass

    def to(self, device: torch.device):
        """
        Move all scheduler tensors to the specified device.

        Args:
            device (torch.device): Target device.
        Returns:
            self
        """
        self.device = device
        for attr in [
            "betas",
            "alphas",
            "alpha_cum_prod",
            "sqrt_alpha_cum_prod",
            "sqrt_one_minus_alpha_cum_prod",
        ]:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(
        self, original: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to the original input according to the forward diffusion process.

        Args:
            original (torch.Tensor): Input tensor of shape (B, C, H, W).
            noise (torch.Tensor): Noise tensor of shape (B, C, H, W).
            t (torch.Tensor): Timesteps tensor of shape (B,).

        Returns:
            torch.Tensor: Noisy sample at timestep t.
        """
        if (
            self.sqrt_alpha_cum_prod is None
            or self.sqrt_one_minus_alpha_cum_prod is None
        ):
            raise ValueError(
                "Scheduler parameters not initialized. Call set_timesteps() first."
            )

        original_shape = original.shape
        batch_size = original_shape[0]

        if t.shape[0] != batch_size:
            raise ValueError(
                f"Timestep tensor batch size ({t.shape[0]}) must match input batch size ({batch_size})"
            )

        # Extract coefficients for the given timesteps
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].to(original.device)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].to(
            original.device
        )

        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Forward diffusion equation: sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    @abstractmethod
    def sample_prev_timestep(
        self, xt: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor
    ) -> tuple:
        """
        Abstract method to sample the previous timestep in the reverse diffusion process.

        Args:
            xt (torch.Tensor): Current noisy sample of shape (B, C, H, W).
            noise_pred (torch.Tensor): Predicted noise of shape (B, C, H, W).
            t (torch.Tensor): Current timestep of shape (B,).

        Returns:
            tuple: (xt-1 sample, x0 estimate)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(num_timesteps={self.num_timesteps})"
