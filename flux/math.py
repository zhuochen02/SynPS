import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, pe_cross: Tensor, info=None) -> Tensor:
    """
    Attention with KV replacement from batch 0 to batch 1 at specific timesteps for editing.
    """
    SPECIAL_BLOCKS = {0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56}

    if info['block_id'] in SPECIAL_BLOCKS:
        q_pe, k_pe = apply_rope(q, k, pe)
        q_pe_cross, k_pe_cross = apply_rope(q, k, pe_cross)

        # Batch 0: standard attention with original pe
        x0 = torch.nn.functional.scaled_dot_product_attention(q_pe[0:1], k_pe[0:1], v[0:1])

        # Batch 1: cross attention using pe_cross (target text + source image)
        k1_txt = k_pe_cross[1:2, :, :512, :]
        k0_img = k_pe_cross[0:1, :, 512:, :]
        q1 = q_pe_cross[1:2]

        v1_txt = v[1:2, :, :512, :]
        v0_img = v[0:1, :, 512:, :]

        k_special = torch.cat([k1_txt, k0_img], dim=2)
        v_special = torch.cat([v1_txt, v0_img], dim=2)

        x1_special = torch.nn.functional.scaled_dot_product_attention(q1, k_special, v_special)

        x = torch.cat([x0, x1_special], dim=0)
    else:
        q_pe, k_pe = apply_rope(q, k, pe)
        x = torch.nn.functional.scaled_dot_product_attention(q_pe, k_pe, v)

    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
