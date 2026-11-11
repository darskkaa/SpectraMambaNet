"""FCC-Net decoder modules, ported from the published reference implementation
(Tang & Zhou et al., "Frequency-enhanced Contextual Conversion Network",
Pattern Recognition 171 (2026) 112235). Math is kept identical to the tested
source; only dead/commented code was stripped and the inline FCM/decoder wiring
was packaged into `FCCDecoder`.

One deliberate deviation: the reference reuses a single ConvBNReLU across call
sites whose activation statistics differ (3 sites in `CCM`, 2 in `FCCDecoder`),
which forces one set of BatchNorm running statistics to cover all of them and
shows up as a train/eval output shift. Those norms are independent by default
here; `share_bn=True` restores the reference form for ablation. See `CCM`.

All ops are Conv/AvgPool/Upsample/Sigmoid — no FFT/DWT — so the whole decoder is
fp16-safe on Turing (T4). Frequency decoupling is done via Octave Convolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rfp import SoftDilationMix


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.relu is not None else x


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, dilation=1, group=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=group)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Octave convolution (frequency decoupling) — FCM building block
# ---------------------------------------------------------------------------
class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = nn.Conv2d(in_channels, int(alpha * in_channels), kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_channels, in_channels - int(alpha * in_channels), kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)
        X_h2l = self.h2g_pool(x)
        X_h = self.h2h(x)
        X_l = self.h2l(X_h2l)
        return X_h, X_l


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.stride = stride
        self.l2l = nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels), kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels), kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels), kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_channels - int(alpha * in_channels), out_channels - int(alpha * out_channels), kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x
        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
        X_h2l = self.h2g_pool(X_h)
        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)
        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l
        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.l2h = nn.Conv2d(int(alpha * out_channels), out_channels, kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(out_channels - int(alpha * out_channels), out_channels, kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x
        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')
        X_h = X_h2h + X_l2h
        return X_h2h, X_l2h, X_h


class Octave(nn.Module):
    """Decouple a feature into high-freq / low-freq / fused (H,L,F) streams.

    `fir` and `mid1` both run at in_channels width — `FirstOctaveConv` never reads its
    `out_channels` argument and `mid1` is `OctaveConv(in_ch, in_ch)` — so 18*in_ch^2 of the
    block's parameters are set by the *encoder* stage width, not by `out_channels`. At
    in_ch=512 that is 4.72 M on an 11x11 map. `dup_mid1=False` drops the second, weight-tied
    application of `mid1`; it costs no parameters and 0.44 GMACs, and `mid2` spans the same
    function class.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dup_mid1=True):
        super().__init__()
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)
        self.dup_mid1 = dup_mid1

    def forward(self, x):
        x_h, x_l = self.fir(x)
        x_h, x_l = self.mid1((x_h, x_l))
        if self.dup_mid1:
            x_h, x_l = self.mid1((x_h, x_l))
        x_h, x_l = self.mid2((x_h, x_l))
        x_ret_h, x_ret_l, x_ret = self.lst((x_h, x_l))
        return x_ret_h, x_ret_l, x_ret  # high, low, fused


class ConcatHLConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x_h, x_l):
        return self.conv1(torch.cat((x_h, x_l), dim=1))


class ConcatHighConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x2, x3, x4):
        x3 = F.interpolate(x3, size=(x2.size(2), x2.size(3)), mode='bilinear')
        x4 = F.interpolate(x4, size=(x2.size(2), x2.size(3)), mode='bilinear')
        return self.conv1(torch.cat((x2, x3, x4), dim=1))


# ---------------------------------------------------------------------------
# SCF — Scale-induced Cross-level Fusion
# ---------------------------------------------------------------------------
class SCF(nn.Module):
    """Third site of the same shared-BatchNorm deviation documented on `CCM`: the reference
    runs all four dilation branches' inputs through one `conv2`, so one set of running stats
    covers four activations that differ by construction (they are different quarters of a
    split, gated by a cascade of sigmoids). `--fcc-share-bn` did not reach here at all until
    now; un-shared is the default, matching `CCM` and `FCCDecoder.conv_h`.
    """

    def __init__(self, enc_channels, channels=64, dilation=(1, 3, 5, 7), share_bn=False):
        super().__init__()
        self.scale_expand = ConvBNReLU(enc_channels, channels * 4, kernel_size=1, stride=1, padding=0)
        self.hidden_conv1 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[0], dilation=dilation[0])
        self.hidden_conv2 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[1], dilation=dilation[1])
        self.hidden_conv3 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[2], dilation=dilation[2])
        self.hidden_conv4 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[3], dilation=dilation[3])
        _conv2 = ConvBNReLU(enc_channels * 2, channels, 1, stride=1, padding=0)
        self.conv2 = nn.ModuleList(
            [_conv2] * 4 if share_bn
            else [_conv2] + [ConvBNReLU(enc_channels * 2, channels, 1, stride=1, padding=0)
                             for _ in range(3)]
        )
        self.conv_out2 = ConvBNReLU(channels * 4, channels, 1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, x_b):  # x_t: high-level, x_b: low-level
        size = x_b.shape[2:]
        x_t_expand = self.scale_expand(x_t)
        x_b_expand = self.scale_expand(x_b)
        xt_split = torch.split(x_t_expand, x_t_expand.size(1) // 4, dim=1)
        xb_split = torch.split(x_b_expand, x_b_expand.size(1) // 4, dim=1)
        xt_up = [F.interpolate(xt_split[i], size=size, mode='bilinear') for i in range(4)]
        x_cat = [torch.cat((xb_split[i], xt_up[i]), dim=1) for i in range(4)]
        f1 = self.conv2[0](x_cat[0])
        f2 = self.conv2[1](x_cat[1])
        f3 = self.conv2[2](x_cat[2])
        f4 = self.conv2[3](x_cat[3])
        r1 = self.hidden_conv1(f1)
        r2 = self.hidden_conv2(f2 * self.sigmoid(r1) + f2)
        r3 = self.hidden_conv3(f3 * self.sigmoid(r2) + f3)
        r4 = self.hidden_conv4(f4 * self.sigmoid(r3) + f4)
        r = torch.cat((r1, r2, r3, r4), dim=1)
        return self.conv_out2(r)


# ---------------------------------------------------------------------------
# CCM — Contextual Conversion Module (edge-guided decoder fusion)
# ---------------------------------------------------------------------------
class CCM(nn.Module):
    """`share_bn=True` reproduces the reference, which routes all three of its
    ConvBNReLU calls through one module. Those three sites are fed a product
    (x_up*x_down), a sum of three tensors, and a gated sum; their measured pre-BN
    variances are 0.322 / 0.965 / 0.363, a 3.0x spread that one set of BatchNorm
    running statistics has to cover. Train mode hides it — each call normalises with
    its own batch statistics — and eval mode does not, so the module's output moves
    when it is switched to eval: 0.330 sigma with the shared norm against 0.006 with
    independent ones (the latter is batch-estimation noise), measured at matched
    weights/data/warm-up. Three CCMs are stacked in the decoder cascade.

    Un-shared is the default; the flag keeps the reference form reproducible as an
    ablation. Cost of un-sharing: 2 extra ConvBNReLU per CCM.
    """

    def __init__(self, channel, share_bn=False):
        super().__init__()
        first = ConvBNReLU(channel, channel, 3, stride=1, padding=1)
        self.conv_1 = nn.ModuleList(
            [first] * 3 if share_bn
            else [first, ConvBNReLU(channel, channel, 3, stride=1, padding=1),
                  ConvBNReLU(channel, channel, 3, stride=1, padding=1)]
        )
        self.conv_2 = BasicConv2d(2 * channel, channel, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_up, x_down, edge):
        if x_down.size()[2:] != edge.size()[2:]:
            edge = F.interpolate(edge, x_down.size()[2:], mode='bilinear', align_corners=False)
        s_wei = self.sigmoid(self.avg_pool(self.conv_1[0](x_up * x_down)))
        x_concat = self.conv_2(torch.cat((x_up * s_wei, x_down * s_wei), dim=1))
        s = self.conv_1[1](x_up + x_down + x_concat)
        s_out = self.conv_1[2](s * edge + s)
        return s_out


# ---------------------------------------------------------------------------
# Boundary-region calibration — adapted from HGM's BDFM idea (erosion/dilation
# region split of a coarse guide map; Zhu et al., "Hybrid Graph Mamba", MICCAI
# 2025), reimplemented with pooling-based morphology instead of GCN so it stays
# fp16-safe and adds no new dependency.
# ---------------------------------------------------------------------------
class RegionPrototypeAttention(nn.Module):
    """Re-project features through per-region prototypes — HGM's BDFM (Zhu et al., MICCAI
    2025, lib/pvt.py:490-527) rewritten as a differentiable GPU op.

    The reference chain is `proto = fbu @ feat` -> `attn = proto @ conv(feat)` ->
    `ctx = proto^T @ attn`, where `fbu` is its {eroded, background, dilated} mask stack
    built by a per-image cv2 erode/dilate loop on the CPU. Two deliberate deviations:

    - Prototypes are divided by their region's mass. HGM's are raw sums, so they scale with
      H*W (7744 at its H/4 attachment) and overflow fp16 long before they mean anything.
    - The region attention is softmaxed over the regions, making `ctx` a convex combination
      of the three prototypes (the OCR formulation, Yuan et al. 2020) rather than HGM's
      unbounded double matmul.
    """

    def __init__(self, channel, n_regions=3):
        super().__init__()
        self.key = nn.Conv2d(channel, channel, 1, bias=False)
        self.out = nn.Conv2d(channel * 2, channel, 1)
        # Zero-init, same convention as delta_conv below: an untrained context branch must
        # not perturb the edge pathway at step 0.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.last_ctx_abs = None

    def forward(self, feat, region):
        B, C, H, W = feat.shape
        r = region.reshape(B, region.shape[1], H * W)
        f = feat.reshape(B, C, H * W)
        proto = torch.bmm(r, f.transpose(1, 2)) / (r.sum(dim=2, keepdim=True) + 1e-6)
        attn = torch.softmax(torch.bmm(proto, self.key(feat).reshape(B, C, H * W)), dim=1)
        ctx = torch.bmm(proto.transpose(1, 2), attn).reshape(B, C, H, W)
        ctx = self.out(torch.cat([feat, ctx], dim=1))
        self.last_ctx_abs = ctx.detach().abs().mean()
        return feat + ctx


class FullResHead(nn.Module):
    """Decode S1 (H/4) to input resolution by learned upsampling instead of an 8x bilinear.

    Every head in the reference decoder is produced at H/4 or coarser and then interpolated,
    so the finest boundary the network can express is ~4 input px wide. An oracle measurement
    — downsample the GT to SxS, upsample it back, score — caps an 88^2 prediction at 0.9673
    mDice on ETIS and 0.9783 on ColonDB. That ceiling is real but it is not the binding
    constraint: we score 0.79 against it, so this is worth ~0.005, and HD95 is the metric it
    should move, not Dice.

    Deviation from the sketch this implements: the f0 skip is fused at H/4, where f0 and S1
    both natively live, rather than at H/2 where f0 would have to be upsampled to meet a map
    it exists to sharpen. `x0_rfb` (= conv_f1(f0)) is already `channel`-wide and computed on
    every forward regardless, so the skip costs one ConvBNReLU rather than a projection too.

    Measured at channel=64: 117,313 params and 1.388 GMACs at a 352 input — ~9% of the whole
    model's compute for 0.3% of its params. The MACs are the real price, and the split is
    fuse ~0.57 G (a 3x3 over 2*channel at H/4), refine ~0.29 G, the two transposes ~0.25 G
    each. If the cost gate binds, narrow `fuse` first; it is the largest term and the one
    furthest from the output.
    """
    def __init__(self, channel):
        super().__init__()
        c2, c4 = max(channel // 2, 1), max(channel // 4, 1)
        self.fuse = ConvBNReLU(channel * 2, channel, 3, stride=1, padding=1)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(channel, c2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c4, 4, stride=2, padding=1),
            nn.BatchNorm2d(c4), nn.ReLU(inplace=True))
        self.refine = ConvBNReLU(c4, c4, 3, stride=1, padding=1)
        self.out = nn.Conv2d(c4, 1, 1)

    def forward(self, s1, x0, out_size):
        x = self.up2(self.up1(self.fuse(torch.cat((s1, x0), dim=1))))
        # Exact at every legal size — the decoder already rejects anything not divisible by
        # 32, so 4*(H/4) == H and the two stride-2 transposes land on it. Kept as a guard
        # because this head has to survive the 0.75x/1.25x multi-scale steps unchanged.
        if tuple(x.shape[-2:]) != (int(out_size[0]), int(out_size[1])):
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return self.out(self.refine(x))


class BoundaryRegionCalib(nn.Module):
    """Recalibrate the edge *logit* inside the boundary band of a coarse guide map.

    `band_px` is a radius in **input-image** pixels; the max-pool kernel that realises it
    is derived at runtime from the stride of whatever feature map the guide arrives on.
    The previous fixed erode_k=5 / dilate_k=9 were feature-map pixels: at f1 resolution
    (44x44 for a 352 input) that is a ~48-input-pixel band, two-thirds of it outside the
    contour. Any polyp under ~50px across — which is most of ETIS and ColonDB — sat
    entirely inside its own "boundary band", so the calibration applied everywhere on
    exactly the cases that matter.
    """
    def __init__(self, channel, band_px=8, input_size=352, guide_temp=0.25, hf_channels=0,
                 use_proto=False):
        super().__init__()
        self.band_px = band_px
        self.input_size = input_size
        self.guide_temp = guide_temp
        # No normalisation here. The region channels are already calibrated probabilities
        # in [0,1] — `erode ~ 1` means "confidently interior". BatchNorm would rescale them
        # to batch-zero-mean/unit-variance, so that statement would mean something different
        # for every image in the batch; and while the guide is still uncertain the boundary
        # channel is near-constant, so BN's 1/sqrt(var) would amplify its noise during
        # exactly the phase in which delta_conv learns its first non-zero weights.
        self.region_conv = nn.Conv2d(3 + hf_channels, channel, 3, padding=1)
        self.act = nn.GELU()
        # Zero-init so calibration starts as an exact identity — the untrained region
        # features can't perturb the decoder cascade at step 0. Signed (no ReLU) so the
        # module can learn to suppress spurious edges, not just sharpen.
        # Zero-init as a gating device is ControlNet's zero convolution (Zhang, Rao &
        # Agrawala, ICCV 2023) and the ReZero/Fixup/LayerScale family; what is ours here is
        # applying it in *logit* space against a boundary band, not in feature space.
        self.delta_conv = nn.Conv2d(channel, 1, 1)
        nn.init.zeros_(self.delta_conv.weight)
        nn.init.zeros_(self.delta_conv.bias)
        # HGM's BDFM does not stop at a scalar offset — it re-projects the *feature map*
        # through the three region prototypes. Without this the region split only ever
        # reaches the output as a 1-channel logit correction.
        self.proto = RegionPrototypeAttention(channel) if use_proto else None

    def band_kernel(self, feat_h):
        r = max(1, int(round(self.band_px * feat_h / self.input_size)))
        return 2 * r + 1

    def forward(self, guide_logit, edge_logit, hf=None):
        """guide_logit: coarse S_g-style logit map. edge_logit: FCC's edge head logit.
        Returns a recalibrated edge logit (caller applies the sigmoid)."""
        # Detached: the band is a routing signal, and letting the decoder's losses backprop
        # through max-pool into S_g would let them push the segmentation guide to change its
        # *sharpness* rather than its content. Sharpening by a low temperature first makes
        # `dilate - erode` a genuine contour band instead of a local dynamic-range map,
        # which on a soft probability is large wherever the guide is merely uncertain.
        prob = torch.sigmoid(guide_logit.detach() / self.guide_temp)
        k = self.band_kernel(prob.shape[-2])
        # Symmetric kernels, so the band straddles the contour instead of sitting outside it.
        dilate = F.max_pool2d(prob, kernel_size=k, stride=1, padding=k // 2)
        erode = -F.max_pool2d(-prob, kernel_size=k, stride=1, padding=k // 2)
        boundary = (dilate - erode).clamp(0, 1)
        if boundary.shape[2:] != edge_logit.shape[2:]:
            def rs(t):
                return F.interpolate(t, size=edge_logit.shape[2:], mode='bilinear', align_corners=False)
            boundary, dilate, erode = rs(boundary), rs(dilate), rs(erode)
        region = torch.cat([erode, boundary, 1 - dilate], dim=1)
        region_in = torch.cat([region, hf], dim=1) if hf is not None else region
        feat = self.act(self.region_conv(region_in))
        if self.proto is not None:
            # Prototypes are pooled over the three *region* channels only; `hf` is a
            # frequency prior, not a partition of the image, and would not define a region.
            feat = self.proto(feat, region)
        delta = self.delta_conv(feat)
        self.last_band_frac = boundary.detach().mean()
        self.last_abs_delta = (boundary * delta).detach().abs().mean()
        # Calibration in logit space, confined to the boundary band. The previous form was
        # clamp(edge_prob + boundary*delta, 0, 1) on an already-sigmoid'd edge: since the
        # edge target is ~0 over most of the image, delta is driven negative, the sum
        # saturates at 0 and the clamp zeroes the gradient — for delta *and* for the
        # upstream edge head. A logit-space offset is bounded by the downstream sigmoid, so
        # no clamp is needed, nothing saturates, and delta is an interpretable log-odds
        # correction.
        return edge_logit + boundary * delta


# ---------------------------------------------------------------------------
# Full FCC-Net decoder: FCM + SCF + CCM wired together
# ---------------------------------------------------------------------------
class FCCDecoder(nn.Module):
    """Takes the 4 PVTv2 encoder features (channels [64,128,320,512]) and returns
    five logit maps at the input resolution: (S_g, S3, S2, S1, edge).
    S1 is the final segmentation output; edge is deep-supervised with Canny GT.

    `f3` may be Mamba-bottleneck-enhanced before being passed in (our novelty hook)."""
    def __init__(self, enc_channels=(64, 128, 320, 512), channel=64, use_region_calib=True,
                 use_rfp=False, share_bn=False, edge_stride=8, region_proto=False,
                 reduce_octave=False, img_size=352, fullres_head=False):
        super().__init__()
        if edge_stride not in (4, 8):
            raise ValueError(f"edge_stride must be 4 or 8, got {edge_stride}")
        c0, c1, c2, c3 = enc_channels
        # FCM: octave-decouple the three high-level stages (f1,f2,f3).
        #
        # `reduce_octave` puts a 1x1 in front of each block so the octave runs at `channel`
        # width instead of the encoder stage's. See `Octave`: two of its four sub-blocks are
        # sized by in_channels alone, so without this the FCM's cost is fixed by the backbone
        # and `channel` is nearly inert. rfb4_1 at c3=512 is 5.05 M params on an 11x11 map,
        # against 0.26 M for Polyp-PVT's entire decoder on the same encoder.
        self.reduce_octave = reduce_octave
        if reduce_octave:
            self.oct_pre = nn.ModuleList([ConvBNReLU(c, channel, 1, stride=1, padding=0)
                                          for c in (c1, c2, c3)])
            oct_in = (channel, channel, channel)
        else:
            self.oct_pre = None
            oct_in = (c1, c2, c3)
        self.rfb2_1 = Octave(oct_in[0], channel, dup_mid1=not reduce_octave)
        self.rfb3_1 = Octave(oct_in[1], channel, dup_mid1=not reduce_octave)
        self.rfb4_1 = Octave(oct_in[2], channel, dup_mid1=not reduce_octave)
        self.high = ConcatHighConv(channel * 3, channel)
        self.low = ConcatHighConv(channel * 3, channel)
        self.hl_cat = ConcatHLConv(channel * 2, channel)
        # Same shared-BatchNorm issue as CCM, on two sites rather than three: the reference
        # runs both the edge head (on x_h_cat) and the S_g head (on the H+L fusion) through
        # one ConvBNReLU, and their pre-BN variances differ 2.0x. conv_h[0] is the edge head,
        # conv_h[1] the S_g head.
        _conv_h = ConvBNReLU(channel, channel, 3, stride=1, padding=1)
        self.conv_h = nn.ModuleList(
            [_conv_h] * 2 if share_bn
            else [_conv_h, ConvBNReLU(channel, channel, 3, stride=1, padding=1)]
        )
        # channel reduction for the SCF path
        self.conv_f1 = ConvBNReLU(c0, channel, 1, stride=1, padding=0)
        self.conv_f2 = ConvBNReLU(c1, channel, 1, stride=1, padding=0)
        self.conv_f3 = ConvBNReLU(c2, channel, 1, stride=1, padding=0)
        self.conv_f4 = ConvBNReLU(c3, channel, 1, stride=1, padding=0)
        self.scf = SCF(channel, channel, share_bn=share_bn)
        self.scf2 = SCF(channel, channel, share_bn=share_bn)
        self.scf3 = SCF(channel, channel, share_bn=share_bn)
        self.ccm = CCM(channel, share_bn=share_bn)
        self.ccm2 = CCM(channel, share_bn=share_bn)
        self.ccm3 = CCM(channel, share_bn=share_bn)
        self.conv_out = nn.Conv2d(channel, 1, 1)
        self.conv_out_e = nn.Conv2d(channel, 1, 1)
        self.conv_out1 = nn.Conv2d(channel, 1, 1)
        self.conv_out2 = nn.Conv2d(channel, 1, 1)
        self.conv_out3 = nn.Conv2d(channel, 1, 1)
        self.use_region_calib = use_region_calib
        if region_proto and not use_region_calib:
            raise ValueError("region_proto extends BoundaryRegionCalib; it needs "
                             "use_region_calib=True")
        if use_region_calib:
            # input_size is the resolution `band_px` is denominated in, so it has to be the
            # base training resolution rather than a literal 352 — see train.py's
            # edge_dilate_kernel, which tracks --img-size for the same reason.
            self.region_calib = BoundaryRegionCalib(channel, input_size=img_size,
                                                    use_proto=region_proto)
        # RFP sits on x_h_cat — the high-frequency stream feeding the edge head — which is the
        # structural position ARF occupies in FARF-Net. See model/rfp.py for why the published
        # form of that selection rule is not worth copying.
        self.use_rfp = use_rfp
        if use_rfp:
            self.rfp_mix = SoftDilationMix(channel)
        # x_h_cat lives at f1's resolution, H/8, so the reference edge head's finest
        # expressible stroke is ~8 input px — against a 1-2 px boundary that is a target the
        # head cannot represent, and CCM's `s*edge + s` degenerates toward `s`. edge_stride=4
        # refines it once against f0 (H/4) before the 1x1, halving the floor. Costs one
        # 3x3 at H/4: ~74 k params, ~0.6 GMACs at 352.
        self.edge_stride = edge_stride
        if edge_stride == 4:
            self.edge_refine = ConvBNReLU(channel * 2, channel, 3, stride=1, padding=1)
        # Off by default, and the returned tuple *grows* rather than reorders when it is on:
        # every consumer of this decoder unpacks by position, so a silent reorder is the one
        # failure mode here that would not raise.
        self.fullres = FullResHead(channel) if fullres_head else None

    def forward(self, feats, out_size, rfp=None):
        # The CCM cascade halves S_g with scale_factor=0.5 (which floors) while the SCF
        # branch inherits the encoder's own floored strides. On a size not divisible by 32
        # the two disagree by one pixel and CCM's `x_up * x_down` raises a broadcast error
        # naming only the tensor dims — e.g. 264 -> "16 must match 17", 440 -> "27 vs 28".
        # train.py:_scales snaps the multi-scale rates to multiples of 32, so this is a
        # guard against a changed rate or an eval at native resolution, not a live bug.
        h_in, w_in = int(out_size[0]), int(out_size[1])
        if h_in % 32 or w_in % 32:
            raise ValueError(
                f"FCCDecoder requires an input size divisible by 32; got {h_in}x{w_in}. "
                f"Nearest valid sizes: {32 * round(h_in / 32)}x{32 * round(w_in / 32)}."
            )
        f0, f1, f2, f3 = feats
        x0_rfb = self.conv_f1(f0)
        x1_rfb = self.conv_f2(f1)
        x2_rfb = self.conv_f3(f2)
        x3_rfb = self.conv_f4(f3)
        # FCM
        g1, g2, g3 = ((self.oct_pre[0](f1), self.oct_pre[1](f2), self.oct_pre[2](f3))
                      if self.oct_pre is not None else (f1, f2, f3))
        x2_h, x2_l, _ = self.rfb2_1(g1)
        x3_h, x3_l, _ = self.rfb3_1(g2)
        x4_h, x4_l, _ = self.rfb4_1(g3)
        x_h_cat = self.high(x2_h, x3_h, x4_h)
        if self.use_rfp:
            if rfp is None:
                raise ValueError("use_rfp=True but no frequency-rank prior was passed to "
                                 "FCCDecoder.forward — SpectraMambaNet must compute it from "
                                 "the input image, not from these features (see model/rfp.py).")
            x_h_cat = self.rfp_mix(x_h_cat, rfp)
        x_l_cat = self.low(x2_l, x3_l, x4_l)
        x_e = self.conv_h[0](x_h_cat)
        if self.edge_stride == 4:
            x_e = self.edge_refine(torch.cat(
                (F.interpolate(x_e, size=x0_rfb.shape[2:], mode='bilinear', align_corners=False),
                 x0_rfb), dim=1))
        edge_logit = self.conv_out_e(x_e)
        edge_raw = torch.sigmoid(edge_logit)
        S_g = self.conv_h[1](self.hl_cat(x_h_cat, x_l_cat))
        S_g_out = self.conv_out(S_g)
        # `edge_raw` is what the edge BCE supervises; `edge` is what guides the CCM cascade.
        # Keeping them separate is what makes --no-fcc-region-calib an attributable ablation:
        # previously the calibrated map was both, so toggling the flag changed the decoder
        # guidance *and* the supervision target at once.
        edge = torch.sigmoid(self.region_calib(S_g_out, edge_logit)) if self.use_region_calib else edge_raw
        # SCF cross-level fusion
        p3 = self.scf3(x3_rfb, x2_rfb)
        p2 = self.scf2(x2_rfb, x1_rfb)
        p1 = self.scf(x1_rfb, x0_rfb)
        # CCM decoder cascade
        S_g_guide34 = F.interpolate(S_g, scale_factor=0.5, mode='bilinear')
        S3 = self.ccm3(S_g_guide34, p3, edge)
        S3_out = self.conv_out3(S3)
        S_g_guide23 = F.interpolate(S3, scale_factor=2, mode='bilinear')
        S2 = self.ccm2(S_g_guide23, p2, edge)
        S2_out = self.conv_out2(S2)
        S_g_guide12 = F.interpolate(S2, scale_factor=2, mode='bilinear')
        S1 = self.ccm(S_g_guide12, p1, edge)
        S1_out = self.conv_out1(S1)
        up = lambda t: F.interpolate(t, size=out_size, mode='bilinear', align_corners=False)
        self.last_edge_mean = edge.detach().mean()
        # The edge slot carries the *logit*, not edge_raw. BCE on a post-sigmoid map has a
        # d/dp = -1/p gradient, and this head collapses to p~1e-3 (the failure the GT dilation
        # above was meant to prevent and did not), so that factor reaches ~1e4-1e6 — past fp16
        # max on its way back through up(). GradScaler cannot rescue an already-overflowing
        # unscaled gradient: it just backs off to 0 and the run NaNs (killed W1 at epoch 39).
        # With logits the gradient is (sigmoid(z) - t), bounded by 1.
        out = (up(S_g_out), up(S3_out), up(S2_out), up(S1_out), up(edge_logit))
        if self.fullres is not None:
            # S1 and x0_rfb are both at H/4 and both `channel`-wide; the head decodes the
            # feature, not the already-1-channel logit S1_out.
            out = out + (self.fullres(S1, x0_rfb, out_size),)
        return out
