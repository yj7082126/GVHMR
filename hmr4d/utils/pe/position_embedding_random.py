from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class PositionEmbeddingRandomST(nn.Module):
    """
    Positional encoding using random frequencies for spatiotemporal grids.
    Produces the same channel count as the 2D variant: 2 * num_pos_feats.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # Extend from (x, y) to (x, y, t) while keeping output channels unchanged.
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode coordinates normalized to [0, 1].
        coords shape: (..., 3) where channels are (x, y, t).
        """
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int] | Tuple[int, int]) -> torch.Tensor:
        """
        Generate positional encoding for a dense (T, H, W) grid.
        If (H, W) is provided, T defaults to 1.

        Returns:
            torch.Tensor: (C, T, H, W)
        """
        if len(size) == 2:
            t, h, w = 1, int(size[0]), int(size[1])
        else:
            t, h, w = int(size[0]), int(size[1]), int(size[2])
        device = self.positional_encoding_gaussian_matrix.device

        grid = torch.ones((t, h, w), device=device, dtype=torch.float32)
        t_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        x_embed = grid.cumsum(dim=2) - 0.5

        t_embed = t_embed / max(t, 1)
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, t_embed], dim=-1))  # (T, H, W, C)
        return pe.permute(3, 0, 1, 2)  # (C, T, H, W)

    def forward_with_coords(
        self, coords_input: torch.Tensor, video_size: Tuple[int, int, int] | Tuple[int, int]
    ) -> torch.Tensor:
        """
        Positionally encode non-normalized coords.
        coords_input shape: (..., 3) in (x, y, t) order.
        video_size: (T, H, W) or (H, W) where T defaults to 1.
        """
        coords = coords_input.clone().to(torch.float)
        if len(video_size) == 2:
            t, h, w = 1, int(video_size[0]), int(video_size[1])
        else:
            t, h, w = int(video_size[0]), int(video_size[1]), int(video_size[2])

        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h
        coords[..., 2] = coords[..., 2] / max(t, 1)
        return self._pe_encoding(coords)
