"""Audit of FARF-Net's Adaptive Receptive Field (ARF) partition.

FARF-Net (WACV 2026) released no code, so this is a reimplementation of its ARF rule from
the paper text, run against the five standard test splits:

    E_i(h,w) = LH^2 + HL^2 + HH^2   (Haar DWT of the feature, then local average pooling)
    E_max    = max_{h,w} E_i
    d_i(h,w) = 1  if E > 0.7*E_max        <- fine receptive field, meant for boundaries
               3  if 0.3*E_max < E <= 0.7*E_max
               5  otherwise                <- coarse receptive field

The claim under test is that `d = 1` concentrates on polyp boundaries. What this script
measures, per dataset and per feature stage:

  auc          how well E alone ranks boundary pixels above non-boundary ones. This is the
               ceiling: the information actually present in E before any thresholding.
  d1_boundary  fraction of true boundary pixels assigned d=1 under the published rule.
  d1_backgrnd  the same for non-boundary pixels. If d1_boundary is not clearly above this,
               the partition carries no boundary information regardless of `auc`.
  *_rank       the same two numbers when E_max normalization is replaced by E's within-image
               rank, split at tertiles. Everything else is held identical.

Deterministic: sorted file order, fixed sample count, no RNG anywhere.

    python src/analysis/arf_audit.py
    python src/analysis/arf_audit.py --limit 200 --out docs/analysis/arf_audit.csv
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DATASETS = ('Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300')
# PVTv2-b2 features_only indices. FARF-Net applies EdgeKAN+ARF to x2 and x3, i.e. the
# stride-8 and stride-16 stages; x1 is too shallow and x4 too coarse for a boundary claim.
STAGES = {'pvt_x2': 1, 'pvt_x3': 2}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def haar_hf_energy(x):
    """Single-level Haar DWT high-frequency energy, summed over channels.

    Matches pywt.dwt2(..., 'haar') up to the orthonormal 1/2 factor shared by all three
    detail bands, which cancels in every ratio and rank this script computes.
    """
    a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
    c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
    lh, hl, hh = (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2
    e = (lh ** 2 + hl ** 2 + hh ** 2).sum(1, keepdim=True)
    return F.avg_pool2d(e, 3, stride=1, padding=1)  # the paper's local average pooling


def boundary_band(mask, radius=4):
    """Symmetric band straddling the GT contour: dilate minus erode."""
    k = np.ones((2 * radius + 1,) * 2, np.uint8)
    return ((cv2.dilate(mask, k) - cv2.erode(mask, k)) > 0).astype(np.uint8)


def specular_mask(bgr):
    """Standard highlight detector: bright and desaturated. Colonoscopy glare is the usual
    suspect for setting a global max, so it is measured rather than assumed."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > 0.85 * 255) & (hsv[..., 1] < 0.25 * 255)).astype(np.uint8)


def auc(scores, labels):
    """Mann-Whitney AUC with tied ranks averaged."""
    order = np.argsort(scores, kind='stable')
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    out = np.empty_like(ranks)
    out[order] = ranks
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return (out[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def dilations_maxnorm(e):
    """The published rule, verbatim."""
    hi, lo = 0.7 * e.max(), 0.3 * e.max()
    return np.where(e > hi, 1, np.where(e > lo, 3, 5))


def dilations_rank(e):
    """The published rule with E_max normalization replaced by within-image rank.

    Tertiles are the rank-space equivalent of a 3-way split; no constant is tuned, and the
    result is invariant to any monotone rescaling of E.
    """
    hi, lo = np.percentile(e, 66.6667), np.percentile(e, 33.3333)
    return np.where(e > hi, 1, np.where(e > lo, 3, 5))


def _pairs(root, limit):
    for ip in sorted(glob.glob(os.path.join(root, 'images', '*')))[:limit]:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = glob.glob(os.path.join(root, 'masks', stem + '.*'))
        if mp:
            yield ip, mp[0]


def audit_split(root, encoder, source, size, band_px, limit):
    rows = []
    stage = STAGES.get(source)
    for image_path, mask_path in _pairs(root, limit):
        bgr = cv2.resize(cv2.imread(image_path), (size, size))
        mask = cv2.resize(cv2.imread(mask_path, 0), (size, size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        if mask.sum() == 0:
            continue

        if stage is None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            e = haar_hf_energy(torch.from_numpy(gray)[None, None])[0, 0].numpy()
        else:
            rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            t = rgb.permute(2, 0, 1)[None].float() / 255.0
            with torch.no_grad():
                feats = encoder((t - IMAGENET_MEAN) / IMAGENET_STD)
            e = haar_hf_energy(feats[stage])[0, 0].numpy()

        h = e.shape[0]
        # The band is defined in input pixels and then resampled, so its physical width is
        # the same at every stage — otherwise a coarser stage would silently get a wider band.
        band = cv2.resize(boundary_band(mask, band_px), (h, h), interpolation=cv2.INTER_NEAREST)
        spec = cv2.resize(specular_mask(bgr), (h, h), interpolation=cv2.INTER_NEAREST)
        if band.sum() < 5 or (band == 0).sum() < 5:
            continue

        d_max, d_rank = dilations_maxnorm(e), dilations_rank(e)
        rows.append({
            'auc': auc(e.ravel(), band.ravel()),
            'd1_boundary': (d_max[band == 1] == 1).mean(),
            'd1_backgrnd': (d_max[band == 0] == 1).mean(),
            'd5_all': (d_max == 5).mean(),
            'd1_boundary_rank': (d_rank[band == 1] == 1).mean(),
            'd1_backgrnd_rank': (d_rank[band == 0] == 1).mean(),
            'argmax_on_specular': float(spec[np.unravel_index(e.argmax(), e.shape)] == 1),
            'specular_area': spec.mean(),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='TestDataset/TestDataset')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--sources', nargs='+', default=['image', 'pvt_x2', 'pvt_x3'],
                   help="'image' needs no encoder; pvt_* run pretrained PVTv2-b2.")
    p.add_argument('--size', type=int, default=352)
    p.add_argument('--band-px', type=int, default=4, help='half-width of the GT contour band')
    p.add_argument('--limit', type=int, default=40, help='images per split')
    p.add_argument('--out', default='docs/analysis/arf_audit.csv')
    args = p.parse_args()

    encoder = None
    if any(s in STAGES for s in args.sources):
        import timm
        encoder = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True).eval()

    out_rows = []
    for source in args.sources:
        print(f"\n=== source: {source} "
              f"({'raw image' if source not in STAGES else 'pretrained PVTv2-b2 feature'}) ===")
        print(f"{'dataset':20s} {'n':>4s} {'AUC':>6s} | "
              f"{'published rule: d1 bnd/bg':>26s} | {'rank rule: d1 bnd/bg':>22s} {'lift':>6s}")
        for name in args.datasets:
            root = os.path.join(args.data_root, name)
            rows = audit_split(root, encoder, source, args.size, args.band_px, args.limit)
            if not rows:
                print(f"{name:20s}   -- no usable image/mask pairs under {root}")
                continue
            agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
            lift = agg['d1_boundary_rank'] / max(agg['d1_backgrnd_rank'], 1e-9)
            print(f"{name:20s} {len(rows):4d} {agg['auc']:6.3f} | "
                  f"{agg['d1_boundary'] * 100:11.1f}% / {agg['d1_backgrnd'] * 100:5.1f}%     | "
                  f"{agg['d1_boundary_rank'] * 100:8.1f}% / {agg['d1_backgrnd_rank'] * 100:5.1f}% "
                  f"{lift:6.2f}x")
            out_rows.append({'source': source, 'dataset': name, 'n': len(rows),
                             'lift_rank': lift, **agg})

    if out_rows:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {len(out_rows)} rows to {args.out}")

        pub = np.array([[r['d1_boundary'], r['d1_backgrnd']] for r in out_rows])
        rnk = np.array([[r['d1_boundary_rank'], r['d1_backgrnd_rank']] for r in out_rows])
        print(f"published rule puts d=1 on boundaries more than background in "
              f"{int((pub[:, 0] > pub[:, 1]).sum())}/{len(pub)} cells")
        print(f"rank rule does so in {int((rnk[:, 0] > rnk[:, 1]).sum())}/{len(rnk)} cells")
        print(f"frames whose global E_max sits on a specular highlight: "
              f"{np.mean([r['argmax_on_specular'] for r in out_rows]) * 100:.1f}%")


if __name__ == '__main__':
    main()
