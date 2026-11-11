"""RFP — Rank-normalized Frequency Prior.

FARF-Net (WACV 2026) selects a per-pixel dilation rate from wavelet high-frequency energy,
partitioning at 0.7*E_max and 0.3*E_max. `src/analysis/arf_audit.py` measures what that rule
actually does on the five standard test splits: E is heavy-tailed, so both thresholds land far
out in the tail, ~99% of pixels fall into the coarsest rate, and the published partition puts
the fine rate on boundary pixels *more* often than on background in only 5 of 15
dataset x feature-stage cells — i.e. it is inert, and slightly backwards as often as not.

The signal itself is fine (AUC of E as a boundary predictor: 0.72-0.80 on the input image,
0.57-0.75 on PVTv2 features). Only the normalizer is broken. Two changes here:

  1. Rank, not ratio-to-max. `r` is E's within-image rank in [0,1], so no image-global scalar
     can be hijacked by a specular highlight and no constant needs tuning per dataset. Under
     the same tertile split this recovers the fine rate on 39-83% of boundary pixels against a
     fixed 32% background floor -- positive lift in 15 of 15 cells.
  2. Soft, learned partition. The mixture weights over the dilation rates come from a small
     map of `r` instead of two hardcoded thresholds, so the cut points are learned and the
     whole thing is differentiable.

`r` is computed from the *input image* rather than from deep features: the audit measures a
stronger boundary signal there (AUC 0.72-0.80 vs 0.57-0.75), the encoder having already
discarded some of it. It carries no gradient — it is data, an extra input channel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_hf_energy(x):
    """Single-level Haar detail energy LH^2 + HL^2 + HH^2, summed over channels.

    Kept bit-identical to `analysis/arf_audit.py::haar_hf_energy` so the module consumes
    exactly the quantity the audit measured.
    """
    a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
    c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
    lh, hl, hh = (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2
    e = (lh ** 2 + hl ** 2 + hh ** 2).sum(1, keepdim=True)
    return F.avg_pool2d(e, 3, stride=1, padding=1)


class RankFrequencyPrior(nn.Module):
    """Input image -> per-pixel rank of its Haar high-frequency energy, in [0,1].

    Parameter-free and gradient-free. Two invariances hold exactly, and neither holds for
    FARF-Net's E/E_max:

    - additive: the three detail bands have zero DC, so E is unchanged by `x + c`;
    - multiplicative: `E(a*x) = a^2 * E(x)`, and rank is invariant under any monotone map.

    So the prior does not move under a brightness or contrast change of the input, which is
    most of what separates one endoscope from another.
    """

    def __init__(self, fp32=True):
        super().__init__()
        # Ranks are derived from squared differences; under fp16 autocast the small-energy
        # tail collapses to a mass of ties and the rank field flattens where it matters most.
        self.fp32 = fp32

    @torch.no_grad()
    def forward(self, img):
        if self.fp32:
            img = img.float()
        e = haar_hf_energy(img)
        b, _, h, w = e.shape
        flat = e.reshape(b, -1)
        # Double argsort is the rank. Ties break by index, which is harmless: ties occur in
        # flat regions where every rate is an equally good answer.
        rank = flat.argsort(dim=1).argsort(dim=1).float()
        denom = max(flat.shape[1] - 1, 1)
        return (rank / denom).reshape(b, 1, h, w)


class SoftDilationMix(nn.Module):
    """Per-pixel soft mixture over dilation rates, selected by the frequency-rank prior.

    y = x + gamma * pointwise( sum_k w_k(r) * depthwise_{dilation=k}(x) )

    One 3x3 depthwise kernel is *shared* across the rates, so the rates index a receptive
    field rather than three independent filters -- that is what makes the module ~5k
    parameters instead of ~15k, and it keeps the ablation about the selection rule rather
    than about added capacity.

    `gamma` is a LayerScale (`net.py:46`), so at `gamma=0` the module is exactly the identity
    and `--use-rfp` is bit-identical to `--no-rfp`. Note this is identity w.r.t. the *host*
    network, not w.r.t. a dilation-1 conv: inserting the module must not perturb the model
    it is being measured against.
    """

    def __init__(self, channels, rates=(1, 3, 5), hidden=16, gamma_init=1e-4):
        super().__init__()
        self.rates = tuple(rates)
        self.weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.project = nn.Conv2d(channels, channels, 1)
        self.selector = nn.Sequential(
            nn.Conv2d(1, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, len(self.rates), 1),
        )
        self.gamma = nn.Parameter(gamma_init * torch.ones(channels))
        self.last_weights = None  # telemetry: mean mixture weight per rate

    def forward(self, x, r):
        if r.shape[-2:] != x.shape[-2:]:
            r = F.interpolate(r, size=x.shape[-2:], mode='bilinear', align_corners=False)
        w = torch.softmax(self.selector(r.type_as(x)), dim=1)
        c = x.shape[1]
        mixed = 0
        for i, d in enumerate(self.rates):
            branch = F.conv2d(x, self.weight, padding=d, dilation=d, groups=c)
            mixed = mixed + w[:, i:i + 1] * branch
        self.last_weights = w.detach().mean(dim=(0, 2, 3))
        return x + self.gamma.view(1, -1, 1, 1).type_as(x) * self.project(mixed)
