"""Score a competitor's released prediction maps on our own metric stack.

Published Dice columns are not mutually comparable: authors differ on the threshold
(sweep vs fixed 0.5), the resolution (native GT vs 352x352), the reduction (per-image
mean vs batch-pooled), and whether the GT is re-binarized after resizing. Our own
re-run of Polyp-PVT lands 0.0208 below its published zero-shot mean for exactly this
reason, and that offset is not transferable to any other row.

When a paper releases its prediction maps, none of that has to be guessed. The maps
are the model's actual output; running them through `test.py`'s estimator produces a
number on our axis with no training, no seed, and no GPU involved. This is the only
way we can put a competitor in our table without either trusting their harness or
retraining their model.

Input maps are expected in the PraNet-family release form: one PNG per test image,
at native GT resolution, min-max normalized to uint8 [0,255] (not thresholded).
`--no-renormalize` skips our re-application of min-max for maps already in that form
-- it is idempotent on them, so the default is to apply it and stay uniform with the
estimator we run on ourselves.
"""

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from test import _binary_metrics, _minmax_normalize

# Directory names competitors use for the same split.
SPLIT_ALIASES = {
    "ETIS": "ETIS-LaribPolypDB",
    "ETIS-Larib": "ETIS-LaribPolypDB",
    "ETIS-LaribPolypDB": "ETIS-LaribPolypDB",
    "ClinicDB": "CVC-ClinicDB",
    "CVC-ClinicDB": "CVC-ClinicDB",
    "ColonDB": "CVC-ColonDB",
    "CVC-ColonDB": "CVC-ColonDB",
    "CVC-300": "CVC-300",
    "EndoScene": "CVC-300",
    "Kvasir": "Kvasir",
    "Kvasir-SEG": "Kvasir",
}

ZERO_SHOT = ("CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")


def _resolve_pred_root(pred_root):
    """Accepts a directory or a .zip. Returns (path, cleanup_dir_or_None)."""
    p = Path(pred_root)
    if p.is_dir():
        return p, None
    if p.suffix.lower() != ".zip":
        raise SystemExit(f"--pred-root must be a directory or a .zip, got {p}")
    tmp = tempfile.mkdtemp(prefix="predmaps_")
    with zipfile.ZipFile(p) as z:
        z.extractall(tmp)
    return Path(tmp), tmp


def _find_split_dirs(root):
    """Map canonical split name -> directory of PNGs, at any depth.

    Release archives nest inconsistently (`result_map/CVC-300/`, `preds/ours/ETIS/`),
    so match on directory *name* rather than on a fixed layout.
    """
    found = {}
    for d in sorted(root.rglob("*")):
        if not d.is_dir() or "__MACOSX" in d.parts:
            continue
        canon = SPLIT_ALIASES.get(d.name)
        if canon and any(f.suffix.lower() in (".png", ".jpg", ".tif") for f in d.iterdir() if f.is_file()):
            if canon in found:
                raise SystemExit(f"two directories claim split {canon}: {found[canon]} and {d}")
            found[canon] = d
    return found


def score_split(pred_dir, gt_dir, renormalize=True, per_image=False):
    import py_sod_metrics

    preds = {f.stem: f for f in pred_dir.iterdir()
             if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".tif")}
    gts = {f.stem: f for f in gt_dir.iterdir()
           if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".tif")}

    missing = sorted(set(gts) - set(preds))
    extra = sorted(set(preds) - set(gts))

    # Held by name as well as by the dict: `dice_h.dynamic_results` is the per-image record
    # (one 256-threshold array per step(), in call order) that `--per-image` needs, and
    # get_results() has already averaged it away by the time it returns.
    dice_h = py_sod_metrics.DICEHandler(with_dynamic=True, with_adaptive=False)
    FM = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "dice": dice_h,
            "iou": py_sod_metrics.IOUHandler(with_dynamic=True, with_adaptive=False),
            "fm_beta03": py_sod_metrics.FmeasureHandler(with_dynamic=True, with_adaptive=True, beta=0.3),
        }
    )
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure(alpha=0.5)
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()

    recalls, hd95s, resized = [], [], 0
    stems = sorted(set(preds) & set(gts))

    for stem in stems:
        gt = np.array(Image.open(gts[stem]).convert("L"))
        gt_native = ((gt > 127).astype(np.uint8)) * 255

        pred_img = Image.open(preds[stem]).convert("L")
        if pred_img.size != (gt_native.shape[1], gt_native.shape[0]):
            # Released at 352x352 rather than native. Upsample rather than downsample the
            # GT, so the GT stays the untouched reference the protocol scores against.
            pred_img = pred_img.resize((gt_native.shape[1], gt_native.shape[0]), Image.BILINEAR)
            resized += 1
        pred_u8 = np.array(pred_img)
        if renormalize:
            pred_u8 = _minmax_normalize(pred_u8.astype(np.float32))

        recall, hd95 = _binary_metrics(pred_u8, gt_native)
        recalls.append(recall)
        hd95s.append(hd95)

        FM.step(pred=pred_u8, gt=gt_native)
        WFM.step(pred=pred_u8, gt=gt_native)
        SM.step(pred=pred_u8, gt=gt_native)
        EM.step(pred=pred_u8, gt=gt_native)
        MAE.step(pred=pred_u8, gt=gt_native)

    fm_res = FM.get_results()
    em_res = EM.get_results()["em"]
    em_curve = em_res.get("dynamic", em_res.get("curve"))
    hd95_arr = np.array(hd95s, dtype=float)
    n_undefined = int(np.isnan(hd95_arr).sum())

    out_per_image = None
    if per_image:
        # The reported mDice is mean-over-thresholds of mean-over-images, so the per-image
        # scalar whose mean *is* that number is mean-over-thresholds for one image. Any other
        # choice (Dice@0.5, adaptive) would test a statistic the table does not report.
        dice_i = [float(np.mean(a)) for a in dice_h.dynamic_results]
        assert abs(float(np.mean(dice_i)) - float(fm_res["dice"]["dynamic"].mean())) < 1e-9
        out_per_image = {
            s: {"dice": d, "recall": float(r), "hd95": (None if np.isnan(h) else float(h))}
            for s, d, r, h in zip(stems, dice_i, recalls, hd95s)
        }

    return {
        "mDice": float(fm_res["dice"]["dynamic"].mean()),
        "mIoU": float(fm_res["iou"]["dynamic"].mean()),
        "Sm": float(SM.get_results()["sm"]),
        "meanEm": float(em_curve.mean()),
        "maxEm": float(em_curve.max()),
        "maxFm_b03": float(fm_res["fm_beta03"]["dynamic"].max()),
        "wFm": float(WFM.get_results()["wfm"]),
        "MAE": float(MAE.get_results()["mae"]),
        "Recall": float(np.mean(recalls)),
        "HD95": float(np.nanmean(hd95_arr)) if n_undefined < len(hd95_arr) else float("nan"),
        "HD95_undefined": n_undefined,
        "n_images": len(recalls),
        "n_gt": len(gts),
        "n_missing_preds": len(missing),
        "n_extra_preds": len(extra),
        "n_resized_preds": resized,
        **({"per_image": out_per_image} if per_image else {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True,
                    help="Directory or .zip of released prediction maps, with one "
                         "subdirectory per split (any nesting depth).")
    ap.add_argument("--test-root", default=os.environ.get("TEST_DATA_ROOT", "TestDataset/TestDataset"))
    ap.add_argument("--name", default=None, help="Label for the model, for the JSON and the header.")
    ap.add_argument("--no-renormalize", action="store_true",
                    help="Score the released uint8 exactly as shipped. Only correct if the "
                         "author already applied PraNet's per-image min-max.")
    ap.add_argument("--out", default=None, help="Write results as JSON.")
    ap.add_argument("--per-image", action="store_true",
                    help="Also record each image's Dice/Recall/HD95 under splits[X].per_image, "
                         "keyed by file stem. This is what paired_significance.py consumes; the "
                         "per-image Dice averages to the reported mDice exactly.")
    args = ap.parse_args()

    pred_root, _tmp = _resolve_pred_root(args.pred_root)
    test_root = Path(args.test_root)
    if not test_root.is_dir():
        raise SystemExit(f"--test-root {test_root} does not exist")

    splits = _find_split_dirs(pred_root)
    if not splits:
        raise SystemExit(f"no recognized split directories under {pred_root}; "
                         f"expected one of {sorted(set(SPLIT_ALIASES))}")

    label = args.name or Path(args.pred_root).stem
    print(f"--- {label}: released prediction maps, scored on the canonical PraNet estimator ---")

    results = {}
    for canon in sorted(splits):
        gt_dir = test_root / canon / "masks"
        if not gt_dir.is_dir():
            print(f"  {canon:20s} SKIPPED (no GT at {gt_dir})")
            continue
        r = score_split(splits[canon], gt_dir, renormalize=not args.no_renormalize,
                        per_image=args.per_image)
        results[canon] = r
        line = (f"Dataset: {canon.upper():20s} | mDice: {r['mDice']:.4f} | mIoU: {r['mIoU']:.4f} | "
                f"S-m: {r['Sm']:.4f} | max E-m: {r['maxEm']:.4f} | wFm: {r['wFm']:.4f} | "
                f"MAE: {r['MAE']:.4f} | Recall@0.5: {r['Recall']:.4f} | HD95@0.5: {r['HD95']:.2f}px "
                f"| n={r['n_images']}/{r['n_gt']}")
        for k, msg in (("n_missing_preds", "missing"), ("n_extra_preds", "unmatched"),
                       ("n_resized_preds", "resized"), ("HD95_undefined", "HD95 undefined")):
            if r[k]:
                line += f" [{r[k]} {msg}]"
        print(line)

    zs = [results[s]["mDice"] for s in ZERO_SHOT if s in results]
    if len(zs) == 3:
        print(f"\nZS-3 mean mDice (CVC-300 / ColonDB / ETIS): {np.mean(zs):.4f}")
        results["_ZS3_mDice"] = float(np.mean(zs))

    if args.out:
        Path(args.out).write_text(json.dumps({"model": label, "splits": results}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
