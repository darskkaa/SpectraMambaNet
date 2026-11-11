"""Pixel-level supervised contrastive term, ported from CTNet.

Reference: CTNet-main/model/Contrastive_head.py and
CTNet-main/utils/pixel_contrast_cross_entropy_loss.py, trained at
`loss = loss_P + 0.1 * loss_contrast` (CTNet-main/Training.py). Both classes are
training-only and are never constructed at eval, so the reported params/MACs are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim=64):
        super().__init__()
        # The reference projects a 32-channel post-Translayer feature. Ours would be the raw
        # 512-ch f3, where each ConvTranspose2d(512, 512, 4) is 4.2 M params — so reduce to the
        # decoder width first and keep the head at ~0.24 M.
        self.reduce = nn.Conv2d(dim_in, dim, kernel_size=1, bias=False)
        self.pro = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        # f3 is H/32; three x2 upsamples land at H/4, the resolution the anchors are drawn at.
        return F.normalize(self.pro(self.reduce(x)), p=2, dim=1)


class PixelContrastLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.07,
                 max_samples=1024, max_views=100):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        # X: (B, N, C) embeddings; y_hat: (B, N) ground truth; y: (B, N) prediction.
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        device = X.device

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            # The >max_views filter is what makes the sampling below total; it guarantees
            # num_hard + num_easy > max_views >= n_view for every class kept, so the
            # reference's unreachable `else: raise Exception` branch is dropped rather than
            # ported.
            this_classes = [c for c in torch.unique(this_y)
                            if (this_y == c).sum() > self.max_views]
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = min(self.max_samples // total_classes, self.max_views)
        if n_view < 2:
            # _contrastive divides by (mask.sum(1)), which is n_view * group_size - 1 and
            # would be 0 for a singleton group.
            return None, None

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float, device=device)
        y_ = torch.zeros(total_classes, dtype=torch.float, device=device)

        ptr = 0
        for ii in range(batch_size):
            this_y_hat, this_y = y_hat[ii], y[ii]
            for cls_id in classes[ii]:
                hard = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
                num_hard, num_easy = hard.shape[0], easy.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                elif num_hard >= n_view / 2:
                    num_hard_keep = n_view - num_easy
                else:
                    num_hard_keep = num_hard
                num_easy_keep = n_view - num_hard_keep

                hard = hard[torch.randperm(num_hard, device=device)[:num_hard_keep]]
                easy = easy[torch.randperm(num_easy, device=device)[:num_easy_keep]]
                indices = torch.cat((hard, easy), dim=0).squeeze(1)

                X_[ptr] = X[ii, indices, :]
                y_[ptr] = cls_id
                ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, labels_.transpose(0, 1)).float()

        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.transpose(0, 1)), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(n_view, n_view)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(anchor_num * n_view, device=mask.device).view(-1, 1), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits)
        neg_logits = (exp_logits * neg_mask).sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp_min(1.0)
        return -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()

    def forward(self, feats, target, seg_logit):
        # feats: (B, C, h, w) at H/4. The reference upsamples the embedding to input resolution
        # instead (ContrastCELoss.forward, `scale_factor=4`), which at our width is a ~0.5 GB
        # activation; the anchors are subsampled to <=1024 either way, so the labels are
        # downsampled to the embedding grid instead — what the original ContrastiveSeg does.
        B, _, h, w = feats.shape
        target = target.float().view(B, 1, *target.shape[-2:])
        seg_logit = seg_logit.float().view(B, 1, *seg_logit.shape[-2:])

        labels = F.interpolate(target, size=(h, w), mode='nearest').long().view(B, -1)
        # The reference takes `_, predict = torch.max(seg, 1)` over a 1-channel logit map, so
        # predict is identically 0 and the hard/easy split collapses to a random split within
        # each class. A binary head thresholds the sigmoid instead.
        predict = (F.interpolate(seg_logit, size=(h, w), mode='bilinear',
                                 align_corners=False).sigmoid() > 0.5).long().view(B, -1)

        feats = feats.float().permute(0, 2, 3, 1).contiguous().view(B, h * w, -1)

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        if feats_ is None:
            # No class cleared max_views anywhere in the batch. The reference returns
            # (None, None) here and then dereferences it in _contrastive; an all-background
            # crop would kill the run.
            return feats.new_zeros(())
        return self._contrastive(feats_, labels_)
