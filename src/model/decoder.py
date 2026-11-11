import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torch.fft
from model.wavelet import WaveletDecodeBranch

class SpectralDecodeBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.cutoff = nn.Parameter(torch.tensor(0.25))
        self.sharpness = nn.Parameter(torch.tensor(10.0))
        self.spectral_fc = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )
        self.phase_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self._freq_cache = {}

    def forward(self, x):
        B, C, H, W = x.shape
        F = torch.fft.rfft2(x.float(), dim=(-2, -1), norm='ortho')
        
        # Cache frequency distance grid
        cache_key = (H, W, str(x.device))
        if cache_key not in self._freq_cache:
            fy = torch.fft.fftfreq(H, device=x.device).view(-1, 1)
            fx = torch.fft.rfftfreq(W, device=x.device).view(1, -1)
            self._freq_cache[cache_key] = torch.sqrt(fx**2 + fy**2).unsqueeze(0).unsqueeze(0)
        freq_dist = self._freq_cache[cache_key]
        
        # Sigmoid-based low-pass mask: 1 at center, 0 at high freq
        lp = torch.sigmoid(-self.sharpness * (freq_dist - self.cutoff))
        hp = 1.0 - lp
        
        mag = torch.abs(F)
        # Compress dynamic range with log1p to prevent float16 overflow in Conv2d
        mag_h = torch.log1p(mag * hp)
        mag_l = torch.log1p(mag * lp)
        combined = torch.cat([mag_h, mag_l], dim=1) 
        
        phase = torch.angle(F).detach()
        decoded_mag = self.spectral_fc(combined)
        # Softplus only enforces magnitude positivity (not a bottleneck like sigmoid*5.0
        # would be); the actual log1p inverse is the expm1 applied below, at reconstruction time.
        decoded_mag = F_nn.softplus(decoded_mag)
        
        # Match phase channels to decoded_mag channels via convex combination (phase_proj's
        # softmax'd weights). in_ch always != out_ch for every decoder stage in this model
        # (512->320, 640->128, 256->64, 128->1), so this is the only path ever taken.
        weights = torch.softmax(self.phase_proj.weight.float(), dim=1)  # Explicit fp32 for AMP safety
        real_p = F_nn.conv2d(torch.cos(phase), weights)
        imag_p = F_nn.conv2d(torch.sin(phase), weights)
        # Direct phasor reconstruction to avoid atan2 massive gradient explosions
        # Upcast to float32 so 1e-4 doesn't underflow to 0.0 in float16!
        real_p = real_p.float()
        imag_p = imag_p.float()

        # 1e-4 in sqrt bounds local gradients to max 100, preventing overflow when multiplied by AMP scaler
        norm = torch.sqrt(real_p**2 + imag_p**2 + 1e-4)
        cos_p = real_p / norm
        sin_p = imag_p / norm

        # Reconstruct complex tensor directly
        # Decompress decoded_mag with expm1 before inverse FFT
        decoded_mag_decomp = torch.expm1(torch.clamp(decoded_mag.float(), max=8.0))
        F_decoded = torch.complex(decoded_mag_decomp * cos_p.float(), decoded_mag_decomp * sin_p.float())

        return torch.fft.irfft2(F_decoded, s=(H, W), norm='ortho').to(x.dtype)

def conv_block(in_ch, out_ch, kernel_size=3):
    """3x3 Conv (+BN+ReLU when the output is a feature map).

    A 1-channel output is a raw logit map — BatchNorm would recentre it and ReLU
    would clamp away every negative (background) logit, so the normalization is
    dropped in that case.
    """
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2,
                        bias=(out_ch == 1))]
    if out_ch > 1:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class DualBranchSpectralDecoder(nn.Module):
    """Spatial + spectral branches fused by concatenation and a learned conv.

    Previously `sigmoid(alpha)*spatial + sigmoid(beta)*spectral` with beta
    initialised to -2.0 (sigmoid ~ 0.12). That scalar gate never opened: Adam's
    per-parameter normalization makes it scale-invariant, so the branch stayed
    at ~12% contribution for entire runs. FCC-Net's own ablation (Table 8) also
    finds gated fusion worse than ungated. Concat-then-conv lets the network
    learn per-channel mixing weights instead of one scalar per stage, and has no
    inert initialisation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = max(out_channels, 8)
        self.spatial = conv_block(in_channels, mid)
        self.spectral = SpectralDecodeBranch(in_channels, mid)
        self.fuse = conv_block(2 * mid, out_channels)

    def forward(self, x):
        return self.fuse(torch.cat([self.spatial(x), self.spectral(x)], dim=1))


class TriBranchSpectralDecoder(nn.Module):
    """Spatial + spectral + wavelet, fused by concatenation (see DualBranch note)."""

    def __init__(self, in_channels, out_channels, wave='db1'):
        super().__init__()
        mid = max(out_channels, 8)
        self.spatial = conv_block(in_channels, mid)
        self.spectral = SpectralDecodeBranch(in_channels, mid)
        self.wavelet = WaveletDecodeBranch(in_channels, mid, wave=wave)
        self.fuse = conv_block(3 * mid, out_channels)

    def forward(self, x):
        return self.fuse(torch.cat([self.spatial(x), self.spectral(x), self.wavelet(x)], dim=1))
