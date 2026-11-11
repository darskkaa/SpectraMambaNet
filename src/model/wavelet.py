import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float32):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters.to(device=x.device, dtype=x.dtype), stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters.to(device=x.device, dtype=x.dtype), stride=2, groups=c, padding=pad)


class WaveletDecodeBranch(nn.Module):
    def __init__(self, in_ch, out_ch, wave='db1'):
        super().__init__()
        dec_filters, rec_filters = create_wavelet_filter(wave, in_ch, in_ch)  # in==out: depthwise-safe
        self.register_buffer('dec_filters', dec_filters)
        self.register_buffer('rec_filters', rec_filters)
        # GELU, not ReLU: LH/HL/HH sub-band coefficients are signed (directional detail,
        # oscillate around 0) — ReLU hard-clips ~half of them to exactly 0 at init, which
        # combined with gamma's saturated sigmoid(-2.0) gate doubly kills gradient into
        # this branch. GELU has no hard zero-floor and preserves sign-carrying gradient.
        self.subband_fc = nn.Sequential(
            nn.Conv2d(in_ch * 4, in_ch * 4, 1, groups=in_ch),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate') if (pad_h or pad_w) else x
        Hp, Wp = xp.shape[-2:]

        coeffs = wavelet_transform(xp, self.dec_filters)
        coeffs = coeffs.reshape(B, C * 4, Hp // 2, Wp // 2)
        coeffs = self.subband_fc(coeffs)
        coeffs = coeffs.reshape(B, C, 4, Hp // 2, Wp // 2)

        rec = inverse_wavelet_transform(coeffs, self.rec_filters)
        rec = rec[:, :, :H, :W] if rec.shape[-2:] != (H, W) else rec
        return self.proj(rec)
