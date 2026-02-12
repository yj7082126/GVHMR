import einops
import numpy as np
import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CameraEncoder(nn.Module):
    """Fuse camera-ray Fourier embeddings into image embeddings."""

    def __init__(self, embed_dim, patch_size=14, num_bands=16, max_resolution=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.camera = FourierPositionEncoding(n=3, num_bands=num_bands, max_resolution=max_resolution)

        self.conv = nn.Conv2d(embed_dim + self.camera.channels, embed_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, img_embeddings, K):
        """
        Args:
            img_embeddings: (B, D, H, W)
            K: (B, 3, 3), intrinsics aligned with `img_embeddings` image space.
        """
        B, D, h, w = img_embeddings.shape
        device = img_embeddings.device

        with torch.no_grad():
            points = torch.stack(
                [
                    torch.arange(0, h, 1, device=device).reshape(-1, 1).repeat(1, w),
                    torch.arange(0, w, 1, device=device).reshape(1, -1).repeat(h, 1),
                ],
                -1,
            ).float()  # (h, w, 2): (row, col)
            points = points[..., [1, 0]]  # -> (x, y)
            points = points * self.patch_size + self.patch_size // 2
            points = points.expand(B, h, w, 2).reshape(B, -1, 2)  # (B, N, 2)

            rays = inverse_perspective_projection(points, K, distance=None)  # (B, N, 3)
            rays_embeddings = self.camera(pos=rays)  # (B, N, C_cam)
            rays_embeddings = einops.rearrange(rays_embeddings, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        z = torch.cat([img_embeddings, rays_embeddings], dim=1)
        z = self.norm(self.conv(z))
        return z


class FourierPositionEncoding(nn.Module):
    def __init__(self, n, num_bands, max_resolution):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = [max_resolution] * n

    @property
    def channels(self):
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        encoding_size *= 2  # sin-cos
        encoding_size += num_dims  # concat original coordinates
        return encoding_size

    def forward(self, pos):
        return _generate_fourier_features(pos, num_bands=self.num_bands, max_resolution=self.max_resolution)


def _generate_fourier_features(pos, num_bands, max_resolution):
    b, n = pos.shape[:2]
    device = pos.device

    min_freq = 1.0
    freq_bands = torch.stack(
        [
            torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
            for res in max_resolution
        ],
        dim=0,
    )

    per_pos_features = torch.stack([pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0)
    per_pos_features = per_pos_features.reshape(b, n, -1)
    per_pos_features = torch.cat(
        [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)],
        dim=-1,
    )
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features


def inverse_perspective_projection(points, K, distance=None):
    """
    Args:
        points: (B, N, 2) in pixels
        K: (B, 3, 3) intrinsics
        distance: None or (B, N, 1)
    """
    points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points = torch.einsum("bij,bkj->bki", torch.inverse(K), points)
    if distance is None:
        return points
    return points * distance

