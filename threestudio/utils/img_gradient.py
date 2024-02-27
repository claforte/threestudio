import torch

from .typing import Float, Tensor, Tuple


def compute_image_gradients(
    img: Float[Tensor, "B H W C"],
) -> Tuple[Float[Tensor, "B H W C"], Float[Tensor, "B H W C"]]:
    """Compute image gradients (dy/dx) for a given image."""
    batch_size, height, width, channels = img.shape

    dy = img[:, 1:, ...] - img[:, :-1, ...]
    dx = img[..., 1:, :] - img[..., :-1, :]

    shapey = [batch_size, 1, width, channels]
    dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=1)
    dy = dy.view(img.shape)

    shapex = [batch_size, height, 1, channels]
    dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=2)
    dx = dx.view(img.shape)

    return dy, dx
