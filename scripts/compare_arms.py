import argparse
import json
import sys
from pathlib import Path

import numpy as np

UNSEEN = ("CVC-300", "CVC-COLONDB", "ETIS-LARIBPOLYPDB")
INDOMAIN = ("KVASIR", "CVC-CLINICDB")

# (key, higher_is_better, print format)
METRICS = (
    ("mDice", True, "{:.4f}"),
    ("mIoU", True, "{:.4f}"),
    ("Sm", True, "{:.4f}"),
    ("meanEm", True, "{:.4f}"),
    ("maxEm", True, "{:.4f}"),
    ("maxFm_b03", True, "{:.4f}"),
    ("wFm", True, "{:.4f}"),
    ("MAE", False, "{:.4f}"),
    ("Recall", True, "{:.4f}"),
    ("HD95", False, "{:.2f}"),
)

# Only these are stored per image, so only these can carry a paired test.
PAIRED = (("dice", True), ("recall", True), ("hd95", False))


def load(path):
    d = json.loads(Path(path).read_text())
    return d.get("model", Path(path).stem), d["splits"]


def paired_arrays(a_split, b_split, field):
    pa, pb = a_split.get("per_image"), b_split.get("per_image")
    if not pa or not pb:
        return None, None
    stems = sorted(set(pa) & set(pb))
    xa, xb = [], []
    for s in stems:
        va, vb = pa[s].get(field), pb[s].get(field)
        # hd95 is None when a prediction or GT is empty; drop the pair, not just one side.
        if va is None or vb is None:
            continue
        xa.append(va)
        xb.append(vb)
    return np.asarray(xa, float), np.asarray(xb, float)


def gt_area_fraction(test_root, split):
    # Split keys in the JSON are upper-cased; the directories are not.
    root = Path(test_root)
    hit = next((d for d in root.iterdir()
                if d.is_dir() and d.name.upper() == split.upper()), None)
    if hit is None or not (hit / "masks").is_dir():
        return {}
    try:
        from PIL import Image
    except ImportError:
        return {}
    out = {}
    for m in (hit / "masks").iterdir():
        if m.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
            continue
        a = np.asarray(Image.open(m).convert("L"), dtype=np.float32) / 255.0
        out[m.stem] = float((a > 0.5).mean())
    return out


def stratified(sa, sb, test_root, field="dice", higher=True):
    # Polyp area is the axis the frequency branch is supposed to act on: the h4 attachment
    # exists because small polyps are 15-31 px at 352 (net.py's own comment), and small means
    # high-frequency. So the mechanism predicts an effect concentrated in the smallest bin,
    # which is a stratification of one metric rather than a hunt across many.
    rows = []
    for s in UNSEEN:
        if s not in sa or s not in sb:
            continue
        areas = gt_area_fraction(test_root, s)
        if not areas:
            continue
        pa, pb = sa[s].get("per_image"), sb[s].get("per_image")
        if not pa or not pb:
            continue
        for stem in sorted(set(pa) & set(pb) & set(areas)):
            va, vb = pa[stem].get(field), pb[stem].get(field)
            if va is None or vb is None:
                continue
            rows.append((areas[stem], va, vb))
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    qs = np.array_split(arr, 4)
    out = []
    for i, q in enumerate(qs):
        d = (q[:, 2] - q[:, 1]) if higher else (q[:, 1] - q[:, 2])
        out.append((i, len(q), q[0, 0], q[-1, 0], q[:, 1].mean(), q[:, 2].mean(), d.mean(), d))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Full metric-by-split comparison of two ablation arms. Prints every "
                    "cell, not the favourable ones: the point is to see whether a win is a "
                    "consistent effect or one of ~50 coin flips.")
    ap.add_argument("--a", required=True, help="baseline arm JSON (e.g. runs/abl_A1.json)")
    ap.add_argument("--b", required=True, help="treatment arm JSON (e.g. runs/abl_A0.json)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--test-root", default="TestDataset/TestDataset",
                    help="GT masks, for the polyp-size stratification.")
    args = ap.parse_args()

    name_a, sa = load(args.a)
    name_b, sb = load(args.b)
    splits = [s for s in UNSEEN + INDOMAIN if s in sa and s in sb]
    if not splits:
        sys.exit("no shared splits between the two JSONs")

    print(f"\nA = {name_a}  ({args.a})")
    print(f"B = {name_b}  ({args.b})")
    print("delta = B - A, signed so that + always means B is better\n")

    wins = losses = 0
    for group, label in ((UNSEEN, "UNSEEN"), (INDOMAIN, "IN-DOMAIN")):
        group = [s for s in group if s in splits]
        if not group:
            continue
        print(f"--- {label} " + "-" * 62)
        print(f"{'metric':<12}" + "".join(f"{s[:14]:>16}" for s in group))
        for key, higher, fmt in METRICS:
            row = f"{key:<12}"
            for s in group:
                if key not in sa[s] or key not in sb[s]:
                    row += f"{'—':>16}"
                    continue
                va, vb = sa[s][key], sb[s][key]
                d = (vb - va) if higher else (va - vb)
                if label == "UNSEEN":
                    wins += d > 0
                    losses += d < 0
                row += f"{fmt.format(vb):>9}{d:+7.4f}" if key != "HD95" \
                    else f"{vb:>9.2f}{d:+7.2f}"
            print(row)
        print()

    # ZS-3 is the internal shorthand; recompute rather than trusting a stored field, so a
    # rescored JSON and a monitor JSON cannot silently disagree here.
    zs = {}
    for nm, s in ((name_a, sa), (name_b, sb)):
        vals = [s[k]["mDice"] for k in UNSEEN if k in s]
        zs[nm] = float(np.mean(vals)) if len(vals) == len(UNSEEN) else None
    if all(v is not None for v in zs.values()):
        print(f"ZS-3 mDice: A={zs[name_a]:.4f}  B={zs[name_b]:.4f}  "
              f"delta={zs[name_b] - zs[name_a]:+.4f}\n")

    print(f"Across the {len(UNSEEN)} unseen splits x {len(METRICS)} metrics: "
          f"B better in {wins}, worse in {losses}.")
    print("Under a true null you expect roughly half. A single favourable cell out of ~30 is "
          "not a finding; a metric that moves the same way on all three splits might be.\n")

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("scipy not installed — skipping the paired tests.")
        return

    print("--- paired Wilcoxon, pooled over the unseen splits " + "-" * 27)
    print("Pooling is the point: n~636 matched images resolves effects that a 3-split "
          "mean-of-means cannot.\n")
    for field, higher in PAIRED:
        xs_a, xs_b = [], []
        for s in UNSEEN:
            if s not in sa or s not in sb:
                continue
            pa, pb = paired_arrays(sa[s], sb[s], field)
            if pa is None or len(pa) == 0:
                continue
            xs_a.append(pa)
            xs_b.append(pb)
        if not xs_a:
            print(f"{field:<8} no per-image data — rescore with --per-image")
            continue
        pa, pb = np.concatenate(xs_a), np.concatenate(xs_b)
        diff = (pb - pa) if higher else (pa - pb)
        n_eff = int((diff != 0).sum())
        if n_eff == 0:
            print(f"{field:<8} n={len(pa)}  every pair identical")
            continue
        stat, p = wilcoxon(pb, pa, zero_method="wilcox", alternative="two-sided")
        better = int((diff > 0).sum())
        verdict = ("B BETTER" if diff.mean() > 0 else "B WORSE") if p < args.alpha \
            else "not distinguishable"
        print(f"{field:<8} n={len(pa):<5} mean delta={diff.mean():+.4f}  "
              f"B better on {better}/{len(pa)} images  p={p:.4g}  -> {verdict}")
    print()

    # Per split as well as pooled. The pooled test answers "is B better overall"; a claim about
    # a mechanism usually lives on one split, and ColonDB alone is 380 of the 636 pooled images
    # so a real effect there can be swamped by two smaller splits moving the other way.
    print("--- paired Wilcoxon, per split " + "-" * 47)
    for s in UNSEEN:
        if s not in sa or s not in sb:
            continue
        for field, higher in PAIRED:
            xa, xb = paired_arrays(sa[s], sb[s], field)
            if xa is None or len(xa) == 0:
                continue
            diff = (xb - xa) if higher else (xa - xb)
            if (diff != 0).sum() == 0:
                continue
            _, p = wilcoxon(xb, xa, zero_method="wilcox", alternative="two-sided")
            mark = "*" if p < args.alpha else " "
            print(f"{s:<20}{field:<8} n={len(xa):<5} delta={diff.mean():+8.4f}  "
                  f"p={p:.4g} {mark}")
    print()

    # Catastrophic-failure rate. A mean Dice hides the shape of the distribution: a model can
    # lose on the mean while missing fewer polyps outright, and for screening the count of
    # images the model effectively failed on is the number with clinical meaning.
    da, db = [], []
    for s in UNSEEN:
        if s in sa and s in sb:
            xa, xb = paired_arrays(sa[s], sb[s], "dice")
            if xa is not None and len(xa):
                da.append(xa)
                db.append(xb)
    if da:
        da, db = np.concatenate(da), np.concatenate(db)
        print("--- failure rate over the unseen splits " + "-" * 38)
        print(f"{'threshold':<12}{'A':>10}{'B':>10}{'delta':>10}   (fewer is better)")
        for t in (0.5, 0.6, 0.75, 0.9):
            fa, fb = float((da < t).mean()), float((db < t).mean())
            print(f"Dice < {t:<6}{fa:>10.4f}{fb:>10.4f}{fa - fb:>+10.4f}")
        print()

    strat = stratified(sa, sb, args.test_root)
    if strat:
        print("--- Dice by polyp size, pooled over the unseen splits " + "-" * 24)
        print("Quartiles of GT area fraction. The frequency axis acts on scale content, and "
              "the h4\nattachment was chosen for 15-31 px polyps, so the mechanism predicts "
              "an effect here.\n")
        print(f"{'quartile':<14}{'n':>6}{'area range':>20}{'A':>10}{'B':>10}{'delta':>10}")
        names = ("Q1 smallest", "Q2", "Q3", "Q4 largest")
        for i, n, lo, hi, ma, mb, md, _ in strat:
            print(f"{names[i]:<14}{n:>6}{f'{lo:.4f}-{hi:.4f}':>20}"
                  f"{ma:>10.4f}{mb:>10.4f}{md:>+10.4f}")
        try:
            from scipy.stats import wilcoxon as _w
            d_small = strat[0][7]
            if (d_small != 0).sum():
                _, p_s = _w(d_small, np.zeros_like(d_small), zero_method="wilcox")
                print(f"\nsmallest quartile: mean delta {d_small.mean():+.4f}, p={p_s:.4g}")
        except ImportError:
            pass
        print()


if __name__ == "__main__":
    main()
