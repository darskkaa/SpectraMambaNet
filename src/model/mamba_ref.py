"""Pure-PyTorch drop-in for `mamba_ssm.modules.mamba_simple.Mamba`.

mamba-ssm ships no Windows wheels (26 releases on PyPI, zero win_amd64), and building it
from source needs MSVC + the CUDA Toolkit against a Store-Python sandbox. This module is
the same math as the CUDA kernel's own reference path (`selective_scan_ref` in
mamba_ssm/ops/selective_scan_interface.py), with **identical parameter names and shapes**:

    in_proj  Linear(d_model, 2*d_inner, bias=bias)
    conv1d   Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv-1)
    x_proj   Linear(d_inner, dt_rank + 2*d_state, bias=False)
    dt_proj  Linear(dt_rank, d_inner, bias=True)
    A_log    (d_inner, d_state)
    D        (d_inner,)
    out_proj Linear(d_inner, d_model, bias=bias)

so a checkpoint trained with the real kernel loads here with zero missing/unexpected keys
and reproduces its scores. Inference only — it is ~50x slower than the fused kernel, and
`train.py` still refuses to launch when `import mamba_ssm` fails.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Cap on the elements of the (b, d, k, k, n) intra-chunk decay tensor. 32M floats = 128 MB,
# which fits an 8 GB card alongside the encoder activations.
_DECAY_BUDGET = 32_000_000


def selective_scan(u, delta, A, B, C, D, z, delta_bias):
    """h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t ;  y_t = C_t h_t + D u_t ;  y *= silu(z).

    Chunked rather than step-by-step: within a chunk the recurrence is solved in closed
    form by a lower-triangular decay matrix `exp(s_k - s_j)`, and only the chunk boundaries
    run sequentially. Every entry of that matrix is a product of `exp(dt*A)` terms with
    A < 0, so it is bounded by 1 and underflows to 0 — which is the correct limit. The
    cumprod-and-divide formulation would instead divide by a number that underflows, so it
    is not used. chunk=1 degenerates to the plain sequential recurrence.
    """
    b, d, l = u.shape
    n = A.shape[1]

    delta = F.softplus(delta + delta_bias[None, :, None])
    dA = torch.einsum('bdl,dn->bdln', delta, A)
    dBu = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    chunk = int(math.sqrt(max(_DECAY_BUDGET // max(b * d * n, 1), 1)))
    chunk = max(1, min(chunk, l))

    h = u.new_zeros(b, d, n)
    ys = []
    for start in range(0, l, chunk):
        k = min(chunk, l - start)
        s = torch.cumsum(dA[:, :, start:start + k], dim=2)          # (b, d, k, n)
        # Mask before the exp, not after: the j > k half of the difference is positive and
        # overflows to +inf, and inf * 0 is NaN.
        logdecay = (s.unsqueeze(3) - s.unsqueeze(2)).masked_fill(
            torch.triu(torch.ones(k, k, device=u.device, dtype=torch.bool), diagonal=1)[
                None, None, :, :, None], float('-inf'))
        decay = torch.exp(logdecay)                                 # (b, d, k, k, n)
        h_all = (torch.einsum('bdkjn,bdjn->bdkn', decay, dBu[:, :, start:start + k])
                 + torch.exp(s) * h.unsqueeze(2))
        ys.append(torch.einsum('bdkn,bnk->bdk', h_all, C[:, :, start:start + k]))
        h = h_all[:, :, -1]

    y = torch.cat(ys, dim=2) + u * D[None, :, None]
    if z is not None:
        y = y * F.silu(z)
    return y


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states, **kwargs):
        b, l, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)            # (b, 2*d_inner, l)
        x, z = xz.chunk(2, dim=1)

        x = F.silu(self.conv1d(x)[..., :l])

        x_dbl = self.x_proj(x.transpose(1, 2))                      # (b, l, dt_rank + 2n)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.reshape(b * l, self.dt_rank).t()
        dt = dt.reshape(self.d_inner, b, l).permute(1, 0, 2)        # (b, d_inner, l)

        y = selective_scan(x, dt, -torch.exp(self.A_log.float()),
                           B.transpose(1, 2), C.transpose(1, 2),
                           self.D.float(), z, self.dt_proj.bias.float())
        return self.out_proj(y.transpose(1, 2))
