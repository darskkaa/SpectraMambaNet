"""Polyp-PVT (Dong et al., CAAI AIR 2023) reference baseline.

Faithful port of Polyp-PVT-main/Polyp-PVT-main/lib/pvt.py with one substitution:
the manual `./pretrained_pth/pvt_v2_b2.pth` load is replaced by timm's
`pvt_v2_b2` (features_only), so this baseline and SpectraMambaNet share the
exact same pretrained encoder weights and there is no encoder confound in the
comparison. `encoder_name` accepts the other B-variants (B2-B5 share
embed_dims), giving the capacity-matched baseline arm for a B3 system.

Everything else is reproduced as-published, including two quirks that are load
bearing for reproducing their 0.808 ColonDB number and would be "bugs" if you
were writing this fresh:
  * BasicConv2d declares a ReLU it never applies (conv -> bn only).
  * SAM softmaxes its 32-channel guidance map over the channel axis and keeps
    channel 1 as the attention weight.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # Declared but not applied, matching the reference implementation.
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class CFM(nn.Module):
    """Cascaded Fusion Module — multiplicative top-down aggregation over the 3 deepest levels."""

    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (self.conv_upsample2(self.upsample(self.upsample(x1)))
                * self.conv_upsample3(self.upsample(x2)) * x3)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        return self.conv4(x3_2)


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        return self.relu(self.conv2(h))


class SAM(nn.Module):
    """Similarity Aggregation Module — graph reasoning over edge-masked anchors."""

    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, size=x.shape[-2:], mode='nearest')
        n, c, h, w = x.size()
        # Channel-axis softmax over the 32-ch guidance map, keeping channel 1 — as published.
        edge = torch.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.softmax(x_proj_reshaped, dim=1)

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state = torch.matmul(x_n_rel, x_proj_reshaped).view(n, self.num_s, *x.size()[2:])
        return x + self.conv_extend(x_state)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class PolypPVT(nn.Module):
    """PVTv2-B2 + CIM + CFM + SAM, two prediction heads.

    Training returns (P1, P2) raw logits at input resolution. Eval returns their
    SUM, which is the reference implementation's test-time convention
    (Polyp-PVT-main/Test.py:38, `F.upsample(res + res1, ...)`) — the two heads
    are an implicit 2-model ensemble, not a discardable auxiliary.
    """

    def __init__(self, channel=32, pretrained_encoder=True, encoder_name='pvt_v2_b2'):
        super().__init__()
        self.backbone = timm.create_model(encoder_name, pretrained=pretrained_encoder,
                                          features_only=True)
        enc = self.backbone.feature_info.channels()  # [64, 128, 320, 512]

        self.Translayer2_0 = BasicConv2d(enc[0], channel, 1)
        self.Translayer2_1 = BasicConv2d(enc[1], channel, 1)
        self.Translayer3_1 = BasicConv2d(enc[2], channel, 1)
        self.Translayer4_1 = BasicConv2d(enc[3], channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(enc[0])
        self.sa = SpatialAttention()
        self.SAM = SAM(num_in=channel)

        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        out_size = x.shape[-2:]
        x1, x2, x3, x4 = self.backbone(x)

        # CIM on the finest (stride-4) features
        x1 = self.ca(x1) * x1
        cim_feature = self.sa(x1) * x1

        # CFM over the 3 deepest levels
        cfm_feature = self.CFM(self.Translayer4_1(x4), self.Translayer3_1(x3), self.Translayer2_1(x2))

        # SAM: CIM features (halved to stride-8, matching cfm_feature) guide the aggregation
        T2 = self.Translayer2_0(cim_feature)
        T2 = F.interpolate(T2, size=cfm_feature.shape[-2:], mode='bilinear', align_corners=True)
        sam_feature = self.SAM(cfm_feature, T2)

        # Interpolate to the true input size rather than a hardcoded scale_factor=8, so
        # multi-scale training (0.75x/1.25x) and non-multiple-of-32 inputs stay exact.
        p1 = F.interpolate(self.out_CFM(cfm_feature), size=out_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_SAM(sam_feature), size=out_size, mode='bilinear', align_corners=False)

        if self.training:
            return p1, p2
        return p1 + p2
