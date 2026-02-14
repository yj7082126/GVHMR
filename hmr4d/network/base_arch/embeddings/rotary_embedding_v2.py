import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.cuda.amp import autocast
from typing import Union, Tuple, List


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.
    Returns:
        grid: [dim, ...]
    """
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)
    return grid


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """Reshape frequency tensor for broadcasting."""
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # Batched real-valued frequencies: (B, S, D)
        if freqs_cis[0].ndim == 3:
            if head_first:
                assert freqs_cis[0].shape == (x.shape[0], x.shape[-2], x.shape[-1]), (
                    f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
                )
                shape = [d if i == 0 or i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            else:
                assert freqs_cis[0].shape == (x.shape[0], x.shape[1], x.shape[-1]), (
                    f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
                )
                shape = [d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)

        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1]), (
                f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            )
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1]), (
                f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            )
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        if freqs_cis.ndim == 3:
            if head_first:
                assert freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1]), (
                    f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
                )
                shape = [d if i == 0 or i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            else:
                assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1]), (
                    f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
                )
                shape = [d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        if head_first:
            assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), (
                f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            )
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
                f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            )
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    # backward-compatible behavior with old implementation
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast(enabled=False)
def apply_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
):
    """
    Backward-compatible apply_rotary_emb from rotary_embedding.py.
    """
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], (
        f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    )

    t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t_mid = (t_mid * freqs.cos() * scale) + (rotate_half(t_mid) * freqs.sin() * scale)
    return torch.cat((t_left, t_mid, t_right), dim=-1)


def apply_rotary_emb_qk(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query/key tensors.
    xq, xk expected shape: [B, S, H, D] if head_first=False, or [B, H, S, D] if head_first=True.
    """
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(pos * interpolation_factor, freqs)
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def get_1d_rotary_pos_embed_from_pos(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    1D RoPE from arbitrary position tensor.
    Args:
        dim: rope dimension for this axis (must be even)
        pos: (...,) position values
    Returns:
        use_real=True: (cos, sin), each (..., dim)
        use_real=False: complex tensor (..., dim//2)
    """
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    pos = pos.float()
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=pos.dtype)[: (dim // 2)] / dim))
    freqs = pos[..., None] * (freqs * interpolation_factor)  # (..., dim//2)
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1)  # (..., dim)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1)  # (..., dim)
        return freqs_cos, freqs_sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))

    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list)

    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list)

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    emb = torch.cat(embs, dim=1)
    return emb


def get_nd_rotary_pos_embed_with_frame_indices(
    rope_dim_list,
    frame_indices: torch.Tensor,
    height: int,
    width: int,
    theta=10000.0,
    use_real=True,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """
    ND RoPE for (T,H,W) context with arbitrary temporal positions.
    Args:
        rope_dim_list: [Dt, Dh, Dw], sum == head_dim
        frame_indices: (B, T) or (T,)
    Returns:
        use_real=True: (cos, sin), each (B, T*H*W, sum(rope_dim_list))
    """
    if frame_indices.dim() == 1:
        frame_indices = frame_indices.unsqueeze(0)
    if frame_indices.dim() != 2:
        raise ValueError(f"frame_indices must be (B,T) or (T,), got {tuple(frame_indices.shape)}")

    B, T = frame_indices.shape
    H, W = int(height), int(width)
    device = frame_indices.device
    dtype = torch.float32

    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)

    t_pos = frame_indices.to(device=device, dtype=dtype)[:, :, None, None].expand(B, T, H, W).reshape(B, -1)
    y_axis = torch.arange(H, device=device, dtype=dtype)
    y_pos = y_axis[None, None, :, None].expand(B, T, H, W).reshape(B, -1)
    x_axis = torch.arange(W, device=device, dtype=dtype)
    x_pos = x_axis[None, None, None, :].expand(B, T, H, W).reshape(B, -1)

    pos_per_axis = [t_pos, y_pos, x_pos]
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed_from_pos(
            rope_dim_list[i],
            pos_per_axis[i],
            theta=theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=-1)
        sin = torch.cat([emb[1] for emb in embs], dim=-1)
        return cos, sin
    emb = torch.cat(embs, dim=-1)
    return emb


def get_encoding(d_model, max_seq_len=4096):
    """
    Backward-compatible helper matching rotary_embedding.py output.
    Return: (L, D)
    """
    t = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i, j -> i j", t, freqs)
    freqs = repeat(freqs, "i j -> i (j r)", r=2)
    return freqs


class ROPE(nn.Module):
    """Backward-compatible 1D RoPE wrapper, plus ND utilities via module-level functions."""

    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        encoding = get_encoding(d_model, max_seq_len)
        self.register_buffer("encoding", encoding, False)

    def rotate_queries_or_keys(self, x):
        """
        Args:
            x : (B, H, L, D)
        Returns:
            rotated_x: (B, H, L, D)
        """
        seq_len, d_model = x.shape[-2:]
        assert d_model == self.d_model

        if seq_len > self.max_seq_len:
            encoding = get_encoding(d_model, seq_len).to(x)
        else:
            encoding = self.encoding[:seq_len]

        return apply_rotary_emb(encoding, x, seq_dim=-2)
