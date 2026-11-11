import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import timm
from model.decoder import DualBranchSpectralDecoder, TriBranchSpectralDecoder, conv_block
from model.fcc_modules import FCCDecoder
from model.rfp import RankFrequencyPrior
from model.contrast import ProjectionHead

try:
    from mamba_ssm import Mamba
except ImportError:
    # mamba-ssm has no Windows wheels, so a local box falls back to the pure-PyTorch
    # reference scan. Same parameter names and shapes, same math as the kernel's own
    # `selective_scan_ref`, ~50x slower — fine for inference and figures, and train.py's
    # check in __main__ still refuses to launch a run without the real kernel.
    from model.mamba_ref import Mamba
    print("=" * 70)
    print("WARNING: mamba_ssm not installed — using the pure-PyTorch reference scan")
    print("(model/mamba_ref.py). Numerically equivalent and state-dict compatible, so")
    print("checkpoints load and score correctly, but far slower. Inference only.")
    print("=" * 70)

PVTV2_VARIANTS = ('pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5')


class PVTv2Encoder(nn.Module):
    """PVTv2 hierarchical encoder via timm.

    Outputs 4 multi-scale feature maps:
        f0: (B, 64,  H/4,  W/4)
        f1: (B, 128, H/8,  W/8)
        f2: (B, 320, H/16, W/16)
        f3: (B, 512, H/32, W/32)

    B2-B5 share embed_dims [64, 128, 320, 512] and differ only in depth, so swapping
    variants changes params and compute but not a single decoder channel count.
    """
    def __init__(self, pretrained=True, name='pvt_v2_b2'):
        super().__init__()
        if name not in PVTV2_VARIANTS:
            raise ValueError(f"encoder must be one of {PVTV2_VARIANTS}, got {name!r}")
        self.name = name
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True
        )
        self.out_channels = self.backbone.feature_info.channels()  # [64, 128, 320, 512]
    
    def forward(self, x):
        return self.backbone(x)  # List of 4 tensors

class LayerScale(nn.Module):
    """Per-channel learned residual scale (Touvron et al., CaiT 2021).

    This is the correct version of what the deleted `sigmoid(gate)` init'd at -2.0 was
    reaching for. A sigmoid gate saturates and has to travel ~4 logits to open; Adam moves
    any parameter ~lr per step regardless of gradient magnitude, so at lr=1e-4 that is
    ~40k steps and it never happened. LayerScale is linear: 1e-4 -> 0.1 is ~1k steps, and
    there is no saturation to escape.
    """
    def __init__(self, dim, init=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1).type_as(x)


class TriAxialMambaBottleneck(nn.Module):
    def __init__(self, dim, bidirectional=False, scan='axial', freq_axis=True, ls_init=1e-4,
                 freq_gain='clamp', freq_gain_scale=3.0, freq_gain_shared=False,
                 third_axis='freq'):
        super().__init__()
        if scan not in ('axial', 'raster'):
            raise ValueError(f"scan must be 'axial' or 'raster', got {scan!r}")
        if third_axis not in ('freq', 'raster'):
            raise ValueError(f"third_axis must be 'freq' or 'raster', got {third_axis!r}")
        if freq_gain not in ('clamp', 'tanh'):
            raise ValueError(f"freq_gain must be 'clamp' or 'tanh', got {freq_gain!r}")
        if freq_gain_scale <= 0:
            raise ValueError(f"freq_gain_scale must be > 0, got {freq_gain_scale}")
        # How the signed log-gain is bounded, and how hard. A0 (2026-08-10) ran the original
        # 'clamp' at scale 3.0 and came back *below* the bi-axial ablation on all five splits;
        # its telemetry says why. `last_delta_mag` is the mean |gain| recorded *before* the
        # bound, and at h4 it read 3.876 against a bound of 3.0 — i.e. most bins sat outside
        # the clamp, pinned at a x20 amplify or a /20 suppress, and torch.clamp passes zero
        # gradient there, so a railed bin can never come back. What looked like convergence
        # over the last ten epochs was a dead gradient. 'tanh' saturates smoothly instead, so
        # the bound is soft and every bin keeps a gradient.
        self.freq_gain = freq_gain
        self.freq_gain_scale = float(freq_gain_scale)
        # mag_head emits one gain per bin *per channel*: at h4 that is 88x45xC independent
        # knobs on the deepest pretrained features, which is enough freedom to memorise the
        # imaging fingerprint of the two training centres (scope optics, native resolution,
        # compression) rather than a transferable spectral prior. That is the shape of A0's
        # failure — it transfers to CVC-300 (-0.004) and ColonDB (-0.001) and collapses on
        # ETIS (-0.013), the most distant centre. Sharing the gain across channels collapses
        # it to one curve over frequency, which can express "attenuate this band" but not
        # "this exact per-channel pattern".
        self.freq_gain_shared = freq_gain_shared
        # What the third branch scans. 'freq' is the contribution: the sequence dimension is the
        # rfft2 bin index. 'raster' is its control — the identical branch (same norm_f, mamba_f
        # (+rev), mag_head, spectral_fuse, ls_f, so the *same parameter count*, and the same
        # bounded multiplicative gain that is exactly identity at zero-init) run over a
        # row-major flattening of the spatial map instead. It answers the one question a
        # reviewer will ask of a tri-axial claim: is the gain from the third *axis being
        # frequency*, or merely from there being a third branch? Note the control is if
        # anything favoured — a raster pass over H*W tokens is ~2x the sequence length of the
        # H*(W//2+1) rfft2 grid.
        self.third_axis = third_axis
        # freq_axis=False drops the spectral scan entirely (no norm_f/mamba_f/mag_head/
        # spectral_fuse/ls_f parameters), leaving a bi-axial H+W scan — the ablation that
        # isolates the frequency axis, which is the specific delta over bi-axial spatial
        # scanning (BAMPolyp) and tri-orientated *spatial* scanning (SegMamba).
        self.freq_axis = freq_axis
        # 'axial' scans each row (B*W sequences of length H) and then each column. 'raster'
        # flattens the whole map into one sequence, row-major for the first scan and
        # column-major for the second — HGM's HGMM runs its stack on `x` and again on
        # `x.permute(0,1,3,2)` and sums (lib/Decoder.py:442-481). The difference is not
        # cosmetic: an axial scan's state never crosses rows, so nothing in it can carry
        # context from the top of the polyp to the bottom. Same parameters either way, so
        # the two modes are directly comparable.
        self.scan = scan
        # A Mamba scan is causal: with one forward pass the first token of a row never sees
        # the last. Every competitive vision-Mamba pairs it with a reversed scan — HGM's
        # BiPixelMambaLayer (lib/Decoder.py:73-161) runs `mamba_forw` on the sequence and
        # `mamba_backw` on `torch.flip(..., dims=[1])`, then merges
        # `x_mamba + flip(y_mamba)`. Costs one extra Mamba per axis (+1.69 M params each at
        # dim=512, ~+5.1 M total) for ~0.16 extra GMACs, since the bottleneck runs at H/32.
        self.bidirectional = bidirectional

        self.norm_h = nn.LayerNorm(dim)
        self.mamba_h = Mamba(d_model=dim)

        self.norm_w = nn.LayerNorm(dim)
        self.mamba_w = Mamba(d_model=dim)

        if freq_axis:
            self.norm_f = nn.LayerNorm(dim)
            self.mamba_f = Mamba(d_model=dim)

        if bidirectional:
            self.mamba_h_rev = Mamba(d_model=dim)
            self.mamba_w_rev = Mamba(d_model=dim)
            if freq_axis:
                self.mamba_f_rev = Mamba(d_model=dim)

        if freq_axis:
            # Signed per-bin gain on the log-magnitude spectrum. Zero-init is what makes the
            # spectral branch an exact identity at step 0 (see forward). Same construction as
            # ControlNet's zero convolutions (Zhang, Rao & Agrawala, ICCV 2023) — a new branch
            # added to a pretrained trunk starts at exactly zero contribution so it cannot
            # perturb the trunk before it has learned anything.
            self.mag_head = nn.Linear(dim, 1 if freq_gain_shared else dim)
            nn.init.zeros_(self.mag_head.weight)
            nn.init.zeros_(self.mag_head.bias)

            # Ungated fusion of the spatial (H+W scan) and spectral branches. Concat-then-1x1
            # gives per-channel mixing weights with no near-off initialisation; the residual
            # add keeps the identity path the pretrained encoder features flow through.
            self.spectral_fuse = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
            )

        # BatchNorm's gamma is 1, so `fused` entered the residual at unit variance from
        # step 0 — the over-correction from deleting the -2.0 gate (0.12 -> 1.0 with
        # nothing in between). LayerScale is the in-between.
        # ls_init=1e-4 is CaiT's value for 24-deep stacks. Measured over 107 epochs here it
        # only reaches ~1e-2, i.e. the scan contributes ~1% of the residual while costing
        # 2.41 G of the analytic MAC floor. Raising it is the cheapest test of whether the
        # branch is weak or merely unreachable within the budget.
        self.ls_h = LayerScale(dim, init=ls_init)
        self.ls_w = LayerScale(dim, init=ls_init)
        if freq_axis:
            self.ls_f = LayerScale(dim, init=ls_init)

        self.last_delta_mag = None
        # Fraction of bins whose pre-bound |gain| is at or past freq_gain_scale. Under 'clamp'
        # this is literally the fraction of the spectrum receiving no gradient this step, so a
        # rising series is the branch dying rather than learning; it is the number that would
        # have caught A0 before it burned 1.84 h. Under 'tanh' nothing is gradient-dead, but a
        # high value still means the gain is running against its range.
        self.last_delta_sat = None
        # thop hooks nn.Module ops and traces nothing inside mamba_ssm's selective-scan
        # kernel, so every MACs number this block contributes is invisible to the profiler.
        # Recording the token count the scans actually consumed lets profile_efficiency.py
        # print a floor on what it missed instead of leaving it as an unquantified caveat.
        self.dim = dim
        self.last_scan_tokens = 0

    def _scan(self, axis, seq):
        """One selective scan over `seq` (B, L, C), or the forward+reversed merge."""
        fwd = getattr(self, f'mamba_{axis}')
        if not self.bidirectional:
            return fwd(seq)
        rev = getattr(self, f'mamba_{axis}_rev')
        # Merge before the residual, exactly as HGM does; the LayerScale that follows is
        # init'd at 1e-4, so doubling this branch's magnitude does not disturb
        # identity-at-init.
        return fwd(seq) + torch.flip(rev(torch.flip(seq, dims=[1])), dims=[1])

    def _raster(self, axis, x, column_major):
        """One full-sequence scan over the flattened map, row-major or column-major.

        44x44 at f1 is 1936 tokens in a single sequence versus 44 in an axial one, which is
        the regime a selective scan is actually for — and the regime HGM's +0.029 comes from.
        """
        B, C, H, W = x.shape
        norm = getattr(self, f'norm_{axis}')
        with torch.amp.autocast(device_type='cuda', enabled=False):
            t = x.float()
            seq = (t.permute(0, 3, 2, 1) if column_major else t.permute(0, 2, 3, 1))
            out = self._scan(axis, norm(seq.contiguous().view(B, H * W, C)))
        out = (out.view(B, W, H, C).permute(0, 3, 2, 1) if column_major
               else out.view(B, H, W, C).permute(0, 3, 1, 2))
        return out.contiguous().type_as(x)

    def _bounded_gain(self, out_f):
        """The signed bounded log-gain, shared by both third-branch modes.

        Factored out so the 'raster' control cannot drift from the 'freq' contribution in the
        one place they must stay identical. Bounding is elementwise, so applying it before the
        reshape (here) instead of after is numerically the same as the original inline form.
        """
        delta_mag = self.mag_head(out_f)
        # The only direct probe of the third branch. `ls_f` gates the *fused* spatial+third
        # pair, so a flat `ls_f` cannot distinguish an inert branch from a well-scaled one;
        # this is the signed per-token gain itself, and a value pinned at 0 means mag_head
        # never moved off its zero init. Recorded pre-bound on purpose: post-bound it can
        # never exceed freq_gain_scale and the saturation that sank A0 would be invisible.
        self.last_delta_mag = delta_mag.detach().abs().mean()
        s = self.freq_gain_scale
        self.last_delta_sat = (delta_mag.detach().abs() >= s).float().mean()
        if self.freq_gain == 'tanh':
            return s * torch.tanh(delta_mag)
        return torch.clamp(delta_mag, min=-s, max=s)

    def forward(self, x):
        B, C, H, W = x.shape

        # h and w each consume H*W tokens whichever scan mode is active (an axial pass runs
        # B*W sequences of length H, same total); f runs over the rfft2 grid, H x (W//2+1).
        third_tokens = 0
        if self.freq_axis:
            third_tokens = H * W if self.third_axis == 'raster' else H * (W // 2 + 1)
        self.last_scan_tokens = (2 if self.bidirectional else 1) * (2 * H * W + third_tokens)

        if self.scan == 'raster':
            out_h_spatial = x + self.ls_h(self._raster('h', x, column_major=False))
            out_w_spatial = out_h_spatial + self.ls_w(
                self._raster('w', out_h_spatial, column_major=True))
        else:
            # Height Scan — disable autocast: mamba_ssm's selective-scan CUDA kernel is
            # numerically fragile in fp16, same reasoning applied to every Mamba call in
            # HGM (Zhu et al., MICCAI 2025, lib/Decoder.py), not just FFT-adjacent ones.
            with torch.amp.autocast(device_type='cuda', enabled=False):
                x_h = x.float().permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
                x_h = self.norm_h(x_h)
                out_h = self._scan('h', x_h)
            out_h_spatial = out_h.view(B, W, H, C).permute(0, 3, 2, 1).contiguous().type_as(x)

            # Mid-residual to stabilize sequence scaling
            out_h_spatial = x + self.ls_h(out_h_spatial)

            # Width Scan
            with torch.amp.autocast(device_type='cuda', enabled=False):
                x_w = out_h_spatial.float().permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
                x_w = self.norm_w(x_w)
                out_w = self._scan('w', x_w)
            out_w = out_w.view(B, H, W, C).permute(0, 3, 1, 2).contiguous().type_as(x)
            out_w_spatial = out_h_spatial + self.ls_w(out_w)

        # Bi-axial mode: the block ends at the spatial scans; last_delta_mag stays None so
        # telemetry logs nothing for a branch that does not exist.
        if not self.freq_axis:
            return out_w_spatial

        if self.third_axis == 'raster':
            # The params-matched control. Same norm_f / mamba_f(+rev) / mag_head / bound, and
            # the same *multiplicative* form as the spectral branch — there the gain multiplies
            # each rfft2 bin's magnitude, here it multiplies each spatial token — so zero-init
            # gives exp(0) = 1 and the branch reproduces its input exactly, identically to the
            # freq path. The only thing that changes is what the sequence dimension indexes.
            with torch.amp.autocast(device_type='cuda', enabled=False):
                t = out_w_spatial.float()
                seq = self.norm_f(t.permute(0, 2, 3, 1).contiguous().view(B, H * W, C))
                bounded = self._bounded_gain(self._scan('f', seq).float())
                bounded = bounded.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                out_f_spatial = t * torch.exp(bounded)
            fused = self.spectral_fuse(
                torch.cat([out_w_spatial, out_f_spatial.type_as(out_w_spatial)], dim=1))
            return (out_w_spatial + self.ls_f(fused)).type_as(x)

        # Spectral Scan on the spatially-refined (H+W scanned) features, chaining the
        # tri-axial design intent — disable AMP: cuFFT requires float32 for non-power-of-2
        # spatial dims.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_f32 = out_w_spatial.float()
            F_spec = torch.fft.rfft2(x_f32, norm='ortho')
            mag   = torch.abs(F_spec)
            phase = torch.angle(F_spec).detach()

            # Compress dynamic range with log1p (mirrors decoder.py's scheme) so the
            # Mamba scan sees a stable scale instead of raw FFT magnitude.
            log_mag = torch.log1p(mag)

            _, _, H_f, W_f = mag.shape
            x_f  = log_mag.permute(0, 2, 3, 1).contiguous().view(B, H_f * W_f, C)
            x_f  = self.norm_f(x_f)
            out_f = self._scan('f', x_f).float()

            # A *signed residual on log-magnitude*, not a replacement. The previous form was
            # expm1(softplus(mamba_out)): softplus(0)=0.693 and expm1(0.693)=1.0, so at init
            # every frequency bin's magnitude was set to exactly 1.0 and the input spectrum
            # was discarded — a flat white spectrum reconstructed with the input's phase,
            # i.e. a large structured perturbation of the deepest pretrained feature at step
            # 0. It also had a floor of ~0.69 and so could never attenuate a bin.
            #
            # With mag_head zero-init'd, delta = 0 and expm1(log1p(mag)) == mag exactly, so
            # the branch reconstructs its input bit-for-bit. exp(delta) is a signed per-bin
            # gain, so it can now both amplify and suppress.
            # Bound the *delta*, not the sum: bounding the sum would break identity whenever
            # log_mag alone exceeded the bound. Gain lands in [e^-s, e^s]. 'clamp' is a hard
            # bound with zero gradient outside it; 'tanh' is the same range reached smoothly,
            # so a bin that runs hot is pulled back rather than frozen. A shared gain
            # broadcasts over C.
            C_g = 1 if self.freq_gain_shared else C
            bounded = self._bounded_gain(out_f)
            bounded = bounded.view(B, H_f, W_f, C_g).permute(0, 3, 1, 2).contiguous()
            out_f_decomp = torch.expm1(log_mag + bounded)

            F_fused      = torch.polar(out_f_decomp, phase)
            out_f_spatial = torch.fft.irfft2(F_fused, s=(H, W), norm='ortho').float()

        fused = self.spectral_fuse(torch.cat([out_w_spatial, out_f_spatial.type_as(out_w_spatial)], dim=1))
        return (out_w_spatial + self.ls_f(fused)).type_as(x)


class PFESASkipAttention(nn.Module):
    def __init__(self, in_channels, temperature=0.1):
        super().__init__()
        # Learnable channel pooling to preserve frequency structure
        self.chan_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        # Learnable temperature for scaling to fix weak dynamic range
        self.temp = nn.Parameter(torch.tensor(temperature))

    def forward(self, x, skip):
        # AMP casts Conv2d output back to float16 even with float32 input.
        # Disable autocast for the entire FFT block — cuFFT requires float32
        # when spatial dims are not powers of two (e.g. 22x22 at this scale).
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_f32 = self.chan_proj(x.float()).float()
            F_x = torch.fft.rfft2(x_f32, norm='ortho')

            mag_x   = torch.abs(F_x)
            phase_x = torch.angle(F_x).detach()  # detach: d(angle)/dF explodes on near-zero bins (cf. decoder.py:41)

            log_mag = torch.log1p(mag_x)
            log_mag_min = log_mag.amin(dim=(-2, -1), keepdim=True)
            log_mag_max = log_mag.amax(dim=(-2, -1), keepdim=True)
            mag_norm = (log_mag - log_mag_min) / (log_mag_max - log_mag_min + 1e-5)

            F_fused     = torch.polar(mag_norm, phase_x)
            spatial_map = torch.fft.irfft2(F_fused, s=skip.shape[-2:], norm='ortho')

            spatial_map = torch.clamp(spatial_map, min=-100.0, max=100.0)
            attn = torch.sigmoid(spatial_map / (F_nn.softplus(self.temp.float()) + 0.1))

        out = skip * attn.to(skip.dtype)
        return out

class DummySkipAttention(nn.Module):
    def forward(self, x, skip):
        return skip


class LiteCBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        rd = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, rd, 1), nn.ReLU(inplace=True), nn.Conv2d(rd, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = x.mean((2, 3), keepdim=True)
        mx = x.amax((2, 3), keepdim=True)
        x = x * torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(s))


class BoundaryAttentionModule(nn.Module):
    """BMANet-inspired edge branch: extracts an edge feature/logit from the finest
    encoder feature (f0) and gates the two finest skip connections. Independently
    revertible via SpectraMambaNet(use_boundary_attn=False) (default)."""
    def __init__(self, in_channels, edge_channels=32):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels, 3, padding=1), nn.BatchNorm2d(edge_channels), nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, edge_channels, 3, padding=1), nn.BatchNorm2d(edge_channels), nn.ReLU(inplace=True),
        )
        self.cbam = LiteCBAM(edge_channels)
        self.edge_pred = nn.Conv2d(edge_channels, 1, 1)

    def forward(self, f0):
        feat = self.cbam(self.edge_conv(f0))
        edge_logit = self.edge_pred(feat)
        edge_attn = torch.sigmoid(edge_logit)
        return edge_attn, edge_logit


class SpectraMambaNet(nn.Module):
    """SpectraMamba-Net: PVTv2-B2 Encoder + Tri-Axial Mamba Bottleneck (height, width, and
    spectral scans, chained) + Dual-Branch Spectral Decoder.

    The decoder's spectral branch reconstructs magnitude in the frequency domain — FFT phase
    is preserved but detached (not learned) — and sums it with a plain spatial Conv2d branch:
    a magnitude-domain reconstruction, not full frequency-domain mask reconstruction.
    """
    def __init__(self, use_pfesa=True, use_spectral_decoder=True, use_mamba_bottleneck=True,
                 use_boundary_attn=False, use_wavelet_decoder=False, use_dsra=False, pretrained_encoder=True,
                 use_fcc_decoder=False, fcc_channel=64, fcc_region_calib=True, use_rfp=False,
                 fcc_share_bn=False, ds_mutation=False, mamba_bidirectional=False,
                 edge_stride=8, mamba_stage='bottleneck', encoder_name='pvt_v2_b2',
                 mamba_scan='axial', fcc_region_proto=False, fcc_reduce_octave=False,
                 img_size=352, fullres_head=False, use_contrast_head=False,
                 mamba_freq_axis=True, mamba_ls_init=1e-4, freq_gain='clamp',
                 freq_gain_scale=3.0, freq_gain_shared=False, third_axis='freq'):
        super().__init__()
        if use_contrast_head and not use_fcc_decoder:
            raise ValueError("use_contrast_head appends an embedding to the FCC decoder's "
                             "training tuple; the legacy decoder path never returns it")
        if edge_stride != 8 and not use_fcc_decoder:
            raise ValueError("edge_stride only affects the FCC decoder's edge head; "
                             "it does nothing without use_fcc_decoder=True")
        if mamba_stage not in ('bottleneck', 'h8', 'both', 'h4', 'h4h8'):
            raise ValueError(f"mamba_stage must be 'bottleneck', 'h8', 'both', 'h4' or "
                             f"'h4h8', got {mamba_stage!r}")
        if mamba_scan not in ('axial', 'raster'):
            raise ValueError(f"mamba_scan must be 'axial' or 'raster', got {mamba_scan!r}")
        if fcc_region_proto and not (use_fcc_decoder and fcc_region_calib):
            raise ValueError("fcc_region_proto extends BoundaryRegionCalib; it needs "
                             "use_fcc_decoder=True and fcc_region_calib=True")
        if mamba_stage != 'bottleneck' and not use_fcc_decoder:
            raise ValueError("mamba_stage='h8'/'both'/'h4'/'h4h8' places a scan on f1 and/or f0, "
                             "which only the FCC decoder path consumes; the legacy decoder "
                             "ignores it")
        if use_wavelet_decoder and not use_spectral_decoder:
            raise ValueError("use_wavelet_decoder requires use_spectral_decoder=True")
        if use_rfp and not use_fcc_decoder:
            raise ValueError("use_rfp attaches to the FCC decoder's high-frequency stream; "
                             "it has nowhere to go without use_fcc_decoder=True")
        if fcc_reduce_octave and not use_fcc_decoder:
            raise ValueError("fcc_reduce_octave resizes the FCC decoder's octave blocks; "
                             "the legacy decoder has none")
        if fullres_head and not use_fcc_decoder:
            raise ValueError("fullres_head decodes the FCC decoder's S1 stream; "
                             "the legacy decoder does not produce one")

        # PVTv2 B2-B5 outputs: [64, 128, 320, 512] at strides [4, 8, 16, 32]
        self.encoder = PVTv2Encoder(pretrained=pretrained_encoder, name=encoder_name)
        embed_dims = self.encoder.out_channels  # [64, 128, 320, 512]

        # Where the tri-axial scan attaches. At the H/32 bottleneck a 352 input gives 11x11,
        # so each axial scan is 11 tokens long and a selective scan over 11 tokens is close to
        # a gated linear layer — which is why it is only 0.16 of our 15.68 GMACs. HGM attaches
        # its Mamba at H/4 and H/8 instead (lib/Decoder.py `hgmm02`/`hgmm03`); f1 at H/8 is
        # 44x44 = 1936 tokens and f0 at H/4 is 7744. 'h8' moves the scan there, 'both' keeps
        # the deep one too, 'h4' scans f0 only, 'h4h8' scans both fine stages and drops the
        # deep one — the placement HGM's HGMM actually uses once its inert windowing is
        # stripped out.
        self.mamba_stage = mamba_stage
        if use_mamba_bottleneck:
            deep = mamba_stage in ('bottleneck', 'both')
            gain_kw = dict(freq_gain=freq_gain, freq_gain_scale=freq_gain_scale,
                           freq_gain_shared=freq_gain_shared, third_axis=third_axis)
            self.bottleneck = (TriAxialMambaBottleneck(embed_dims[3],
                                                       bidirectional=mamba_bidirectional,
                                                       scan=mamba_scan,
                                                       freq_axis=mamba_freq_axis,
                                                       ls_init=mamba_ls_init, **gain_kw)
                               if deep else nn.Identity())
            self.mamba_h8 = (TriAxialMambaBottleneck(embed_dims[1],
                                                     bidirectional=mamba_bidirectional,
                                                     scan=mamba_scan,
                                                     freq_axis=mamba_freq_axis,
                                                     ls_init=mamba_ls_init, **gain_kw)
                             if mamba_stage in ('h8', 'both', 'h4h8') else None)
            # f0 at H/4 is 88x88 = 7744 tokens, HGM's only working Mamba placement. The 18
            # ETIS polyps under 0.5% of frame are 15-31 px at 352, i.e. 5.4 cells here
            # against 2.7 at H/8 — the scale at which they are still separable.
            self.mamba_h4 = (TriAxialMambaBottleneck(embed_dims[0],
                                                     bidirectional=mamba_bidirectional,
                                                     scan=mamba_scan,
                                                     freq_axis=mamba_freq_axis,
                                                     ls_init=mamba_ls_init, **gain_kw)
                             if mamba_stage in ('h4', 'h4h8') else None)
        else:
            if mamba_bidirectional:
                raise ValueError("mamba_bidirectional needs use_mamba_bottleneck=True")
            if mamba_stage != 'bottleneck':
                raise ValueError("mamba_stage needs use_mamba_bottleneck=True")
            if mamba_scan != 'axial':
                raise ValueError("mamba_scan needs use_mamba_bottleneck=True")
            if not mamba_freq_axis:
                raise ValueError("mamba_freq_axis=False removes the frequency scan from the "
                                 "tri-axial block; there is no block without "
                                 "use_mamba_bottleneck=True")
            if mamba_ls_init != 1e-4:
                raise ValueError("mamba_ls_init scales the tri-axial block's residual; there "
                                 "is no block without use_mamba_bottleneck=True")
            if freq_gain != 'clamp' or freq_gain_scale != 3.0 or freq_gain_shared:
                raise ValueError("freq_gain/freq_gain_scale/freq_gain_shared bound the "
                                 "spectral branch's per-bin gain; there is no spectral branch "
                                 "without use_mamba_bottleneck=True")
            if third_axis != 'freq':
                raise ValueError("third_axis selects what the block's third branch scans; "
                                 "there is no block without use_mamba_bottleneck=True")
            self.bottleneck = nn.Identity()
            self.mamba_h8 = None
            self.mamba_h4 = None

        # FCC-Net decoder (ported reference): FCM(octave) + SCF + CCM, edge-supervised,
        # 5-head deep supervision. When enabled it fully replaces the gated spectral/DSRA
        # decoder path below. The Mamba bottleneck still runs on f3 first (our novelty hook).
        self.use_fcc_decoder = use_fcc_decoder
        self.use_rfp = use_rfp
        # Set by --ds-mutation. Only legal when the loss supervises every summed subset of the
        # four seg heads (train.py:_fcc_loss); summing independently-supervised heads is not.
        self.ds_mutation = ds_mutation
        # Inference-only, and deliberately not a constructor arg: how many of
        # (S1, S2, S3, S_g) the summed prediction uses. 4 is what every archived run reports.
        # Mutation supervision covers every summed subset, so 1..4 are all legally predictable
        # quantities and src/sweep_heads.py re-scores a finished checkpoint at each.
        self.ds_infer_heads = 4
        if ds_mutation and not use_fcc_decoder:
            raise ValueError("ds_mutation supervises the FCC decoder's four seg heads; "
                             "it has nothing to act on without use_fcc_decoder=True")
        if use_fcc_decoder:
            self.fcc_decoder = FCCDecoder(tuple(embed_dims), channel=fcc_channel,
                                          use_region_calib=fcc_region_calib, use_rfp=use_rfp,
                                          share_bn=fcc_share_bn, edge_stride=edge_stride,
                                          region_proto=fcc_region_proto,
                                          reduce_octave=fcc_reduce_octave, img_size=img_size,
                                          fullres_head=fullres_head)
            # Computed from the input image, not from encoder features: the audit measures a
            # stronger boundary signal there (AUC 0.72-0.80 vs 0.57-0.75).
            self.rfp = RankFrequencyPrior() if use_rfp else None
            # Training-only, and only emitted inside `if self.training`, so profile_efficiency
            # never sees it and the reported cost stays the deployed cost.
            self.contrast_head = (ProjectionHead(embed_dims[3], fcc_channel)
                                  if use_contrast_head else None)
            return

        self.use_boundary_attn = use_boundary_attn
        if use_boundary_attn:
            self.boundary_module = BoundaryAttentionModule(embed_dims[0])  # 64ch, matches f0

        def get_decoder(in_ch, out_ch):
            if use_wavelet_decoder:
                return TriBranchSpectralDecoder(in_ch, out_ch)
            elif use_spectral_decoder:
                return DualBranchSpectralDecoder(in_ch, out_ch)
            else:
                return conv_block(in_ch, out_ch)

        # Decoder chain: 512→320, 640→128, 256→64, 128→1
        self.decoder3 = get_decoder(embed_dims[3], embed_dims[2])         # 512 → 320
        self.decoder2 = get_decoder(embed_dims[2] + embed_dims[2], embed_dims[1])  # 320+320=640 → 128
        self.decoder1 = get_decoder(embed_dims[1] + embed_dims[1], embed_dims[0])  # 128+128=256 → 64
        self.decoder0 = get_decoder(embed_dims[0] + embed_dims[0], 1)              # 64+64=128 → 1

        self.skip_attn2 = PFESASkipAttention(embed_dims[2]) if use_pfesa else DummySkipAttention()
        self.skip_attn1 = PFESASkipAttention(embed_dims[1]) if use_pfesa else DummySkipAttention()
        self.skip_attn0 = PFESASkipAttention(embed_dims[0]) if use_pfesa else DummySkipAttention()
        
        # Deep Supervision Heads
        self.ds3 = nn.Conv2d(embed_dims[2], 1, 1)   # 320 → 1
        self.ds2 = nn.Conv2d(embed_dims[1], 1, 1)   # 128 → 1
        self.ds1 = nn.Conv2d(embed_dims[0], 1, 1)   # 64  → 1

        # DSRA (Dual-Supervised Reverse Attention, PraNet-V2-inspired): background-logit
        # siblings to ds3/ds2/ds1, used to gate the next-finer skip connection with
        # softmax(fg - bg) instead of a single-polarity sigmoid.
        self.use_dsra = use_dsra
        if use_dsra:
            self.ds3_bg = nn.Conv2d(embed_dims[2], 1, 1)   # 320 → 1
            self.ds2_bg = nn.Conv2d(embed_dims[1], 1, 1)   # 128 → 1
            self.ds1_bg = nn.Conv2d(embed_dims[0], 1, 1)   # 64  → 1

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)

        if self.use_fcc_decoder:
            # Mamba bottleneck enriches the deepest feature before it feeds both the
            # frequency-edge (FCM) and cross-level (SCF) paths — the novelty hook.
            f3 = self.bottleneck(f3)
            if self.mamba_h8 is not None:
                # f1 feeds both FCM (rfb2_1 -> x_h_cat, the edge stream) and SCF (conv_f2),
                # so a scan here reaches the boundary pathway rather than only the coarse
                # semantic one the H/32 bottleneck sits on.
                f1 = self.mamba_h8(f1)
            if self.mamba_h4 is not None:
                # f0 is the only skip that reaches S1, the head we actually predict with.
                f0 = self.mamba_h4(f0)
            r = self.rfp(x) if self.use_rfp else None
            dec = self.fcc_decoder([f0, f1, f2, f3], x.shape[-2:], rfp=r)
            S_g, S3, S2, S1, edge = dec[:5]
            # 6th element only when --fullres-head is on; the head appends rather than
            # displacing so the four indices above keep meaning what every archived run,
            # every test and _fcc_loss already assume.
            S_full = dec[5] if len(dec) == 6 else None
            if self.training:
                # (final, aux-fine, aux-mid, aux-coarse, edge-logit[, full-res][, embedding]);
                # loss weights the finer maps higher. edge is a raw logit — BCE-with-logits,
                # see _fcc_loss. The embedding appends at the END, after S_full, so the five
                # leading indices keep meaning what every archived run assumes; _fcc_loss peels
                # it off the tail by `self.contrast_loss is not None`, never by length.
                out = (S1, S2, S3, S_g, edge) if S_full is None \
                    else (S1, S2, S3, S_g, edge, S_full)
                if self.contrast_head is not None:
                    out = out + (self.contrast_head(f3),)
                return out
            if S_full is not None:
                # The full-res head is the prediction, not an auxiliary: it is supervised at
                # weight 1.0 and is the only head that was never bilinearly upsampled. That
                # also removes --ds-mutation's inference-side purpose (it exists to make
                # S1+S2+S3+S_g a legally predictable quantity); it stays valid as a
                # multi-head regulariser, which is what run DS0 measures.
                return S_full
            if self.ds_mutation:
                # Mutation supervision optimises every summed subset of these four, so the
                # full sum is a supervised output rather than an untrained combination.
                # Same rule Polyp-PVT (Test.py:37, P1+P2) and PraNet-V2 (MyTest_med.py:38,
                # p2+p3+p4+p5) publish under; all four are already at input resolution.
                # Polyp-PVT sums two, PraNet-V2 four — the count is a published choice, and
                # summing more inflates confidence, which costs precision on the easy splits.
                heads = [S1, S2, S3, S_g][:self.ds_infer_heads]
                return heads[0] if len(heads) == 1 else sum(heads)
            return S1

        # Bottleneck: Tri-Axial Mamba on deepest features (512-dim)
        d3 = self.bottleneck(f3)
        d3 = self.decoder3(d3)

        fg3 = bg3 = fg2 = bg2 = fg1 = bg1 = None
        if self.use_dsra:
            fg3, bg3 = self.ds3(d3), self.ds3_bg(d3)
            gamma3 = torch.softmax(torch.stack([fg3, bg3], dim=1), dim=1)[:, 0]

        # Stage 2: Upsample + PFESA skip + decode
        d3_clamped = torch.clamp(d3.float(), min=-100.0, max=100.0)
        up2 = nn.functional.interpolate(d3_clamped, size=f2.shape[-2:], mode='bilinear', align_corners=False).to(d3.dtype)
        s2 = self.skip_attn2(up2, f2)
        if self.use_dsra:
            gamma3_up = nn.functional.interpolate(gamma3, size=s2.shape[-2:], mode='bilinear', align_corners=False)
            s2 = s2 * (1.0 + gamma3_up)
        d2 = self.decoder2(torch.cat([up2, s2], dim=1))

        if self.use_dsra:
            fg2, bg2 = self.ds2(d2), self.ds2_bg(d2)
            gamma2 = torch.softmax(torch.stack([fg2, bg2], dim=1), dim=1)[:, 0]

        # Stage 1: Upsample + PFESA skip + decode
        d2_clamped = torch.clamp(d2.float(), min=-100.0, max=100.0)
        up1 = nn.functional.interpolate(d2_clamped, size=f1.shape[-2:], mode='bilinear', align_corners=False).to(d2.dtype)
        s1 = self.skip_attn1(up1, f1)
        if self.use_dsra:
            gamma2_up = nn.functional.interpolate(gamma2, size=s1.shape[-2:], mode='bilinear', align_corners=False)
            s1 = s1 * (1.0 + gamma2_up)

        edge_logit = None
        edge_attn = None
        if self.use_boundary_attn:
            edge_attn, edge_logit = self.boundary_module(f0)
            edge_attn1 = nn.functional.interpolate(edge_attn, size=s1.shape[-2:], mode='bilinear', align_corners=False)
            s1 = s1 * (1.0 + edge_attn1)

        d1 = self.decoder1(torch.cat([up1, s1], dim=1))

        if self.use_dsra:
            fg1, bg1 = self.ds1(d1), self.ds1_bg(d1)
            gamma1 = torch.softmax(torch.stack([fg1, bg1], dim=1), dim=1)[:, 0]

        # Stage 0: Upsample + PFESA skip + final output
        d1_clamped = torch.clamp(d1.float(), min=-100.0, max=100.0)
        up0 = nn.functional.interpolate(d1_clamped, size=f0.shape[-2:], mode='bilinear', align_corners=False).to(d1.dtype)
        s0 = self.skip_attn0(up0, f0)
        if self.use_dsra:
            gamma1_up = nn.functional.interpolate(gamma1, size=s0.shape[-2:], mode='bilinear', align_corners=False)
            s0 = s0 * (1.0 + gamma1_up)
        if self.use_boundary_attn:
            s0 = s0 * (1.0 + edge_attn)  # edge_attn already at f0 res
        out = self.decoder0(torch.cat([up0, s0], dim=1))

        if self.training:
            out3 = fg3 if self.use_dsra else self.ds3(d3)
            out2 = fg2 if self.use_dsra else self.ds2(d2)
            out1 = fg1 if self.use_dsra else self.ds1(d1)
            # DS order: (final output, finest aux, mid aux, coarsest aux)
            # Loss weights expect this order: finer aux gets more weight.
            result = (out, out1, out2, out3)
            if self.use_boundary_attn:
                result = result + (edge_logit,)
            if self.use_dsra:
                result = result + (bg3, bg2, bg1)
            return result

        return out
