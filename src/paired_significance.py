"""Is our Dice actually different from a competitor's, or is it the same model with noise?

The usual answer is "retrain at three seeds and t-test the means", which costs three full runs
and still tests n=3. When the competitor released its prediction maps we can do much better for
free: their maps and ours cover the *same* 798 test images, so every image is a matched pair and
the test is a paired one over n=636 (zero-shot) instead of n=3.

The paired quantity is exactly the number the table reports. `score_pred_maps.py --per-image`
records, for each image, the mean of its Dice curve over the 256 thresholds; the mean of those
per-image values *is* mDice (asserted there). So a significant result here is a statement about
the published column, not about some other statistic that happens to correlate with it.

Wilcoxon signed-rank rather than a paired t-test: per-image Dice is bounded in [0,1] and piles up
near 1.0 on the easy splits, so the differences are not remotely normal. Holm-Bonferroni across
the competitor family within each split, because comparing against eleven models and quoting the
smallest p is how a 0.05 turns into a 0.43.

This measures nothing about training. It compares two fixed sets of maps, so it inherits whatever
seed either model happened to get -- it answers "do these two models differ on these images",
which is the question a reader asks of a comparison table, and not "would a retrain differ".

Usage:
    python src/score_pred_maps.py --pred-root <ours> --per-image --out runs/ours_perimg.json
    python src/paired_significance.py --ours runs/ours_perimg.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ZERO_SHOT = ("CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")

# All eleven released-map scores, Polyp-PVT and CTNet included. Dropping those two would report
# a clean sweep against nine weak competitors.
DEFAULT_OTHERS = ("results/competitors",)


def load(path):
    d = json.loads(Path(path).read_text())
    name = d.get("model", Path(path).stem)
    out = {}
    for split, r in d.get("splits", {}).items():
        if not isinstance(r, dict) or "per_image" not in r:
            continue
        out[split] = {k: v["dice"] for k, v in r["per_image"].items()}
    if not out:
        raise SystemExit(f"{path} has no per_image data — rescore it with "
                         f"`score_pred_maps.py --per-image`")
    return name, out


def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p in the input order."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def paired(ours, theirs, label, split):
    common = sorted(set(ours) & set(theirs))
    only_ours, only_theirs = set(ours) - set(theirs), set(theirs) - set(ours)
    if only_ours or only_theirs:
        # Not fatal -- some releases drop a handful of images -- but a silently shrunken
        # pairing would quietly change which images the comparison is over.
        print(f"  [warn] {label}/{split}: pairing on {len(common)} of "
              f"{len(ours)}/{len(theirs)} ({len(only_ours)} ours-only, "
              f"{len(only_theirs)} theirs-only)")
    a = np.array([ours[s] for s in common])
    b = np.array([theirs[s] for s in common])
    d = a - b
    if not np.any(d):
        return dict(n=len(common), median=0.0, mean=0.0, wins=0, p=1.0)
    stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return dict(n=len(common), median=float(np.median(d)), mean=float(d.mean()),
                wins=int((d > 0).sum()), p=float(stat.pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="per-image JSON for our model")
    ap.add_argument("--others", nargs="+", default=list(DEFAULT_OTHERS),
                    help="per-image JSONs, or directories of them, for the competitors. "
                         "Defaults to all eleven released-map scores.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="Write the full table as JSON.")
    args = ap.parse_args()

    # The defaults are cwd-relative, so running this from anywhere but the repo root used to
    # reach load() with a path that does not exist. FileNotFoundError is not SystemExit, so it
    # skipped the "rescore it" collector below and came out as a traceback; an empty predmaps
    # directory was worse, contributing zero rivals in silence and printing a sweep against the
    # competitors that were never loaded. Both are the same failure the DEFAULT_OTHERS comment
    # above is about, so both are caught here instead.
    files, missing = [], []
    for o in args.others:
        p = Path(o)
        if p.is_dir():
            found = sorted(p.glob("*.json"))
            (files.extend(found) if found else missing.append(f"{o} (directory has no *.json)"))
        elif p.is_file():
            files.append(p)
        else:
            missing.append(f"{o} (no such file or directory)")
    if missing:
        raise SystemExit("cannot read these --others paths:\n  " + "\n  ".join(missing)
                         + f"\nthe defaults are relative to the repo root; cwd is {Path.cwd()}")

    our_name, our = load(args.ours)
    rivals, need_rescore = [], []
    for f in files:
        try:
            rivals.append(load(f))
        except SystemExit as e:
            need_rescore.append(str(e))
    if need_rescore:
        # Report every one at once: rescoring is a slow per-model pass and finding out about
        # them one run at a time is the difference between one afternoon and four.
        raise SystemExit("\n".join(need_rescore))

    # ZS-3 pooled: one test over all three unseen splits, which is the column the paper leads
    # with. Stems can collide across splits, so namespace them before merging. Note this
    # weights each *image* equally, whereas the ZS-3 mDice headline weights each *split*
    # equally (ColonDB is 380 images and CVC-300 is 60), so `mean d` here will not equal the
    # difference of the two ZS-3 numbers. It is the right pooling for a test over images.
    def pool(d):
        return {f"{s}/{k}": v for s in ZERO_SHOT if s in d for k, v in d[s].items()}
    our["ZS-3 (pooled)"] = pool(our)
    for _, d in rivals:
        d["ZS-3 (pooled)"] = pool(d)

    splits = [s for s in ("ZS-3 (pooled)", *ZERO_SHOT, "Kvasir", "CVC-ClinicDB")
              if s in our and any(s in d for _, d in rivals)]

    table = {}
    print(f"Paired per-image Dice, {our_name} vs released maps. Wilcoxon signed-rank, "
          f"two-sided, Holm-corrected within each split.\n")
    for split in splits:
        rows = [(name, paired(our[split], d[split], name, split))
                for name, d in rivals if split in d]
        adj = holm(np.array([r["p"] for _, r in rows]))
        table[split] = {name: {**r, "p_holm": float(pa)}
                        for (name, r), pa in zip(rows, adj)}

        print(f"=== {split} ===")
        print(f"  {'competitor':22s} {'n':>5s} {'median d':>10s} {'mean d':>9s} "
              f"{'we win':>8s} {'p':>10s} {'p(Holm)':>10s}")
        for (name, r), pa in sorted(zip(rows, adj), key=lambda t: -t[0][1]["mean"]):
            mark = "*" if pa < args.alpha else " "
            print(f"  {name:22s} {r['n']:5d} {r['median']:+10.4f} {r['mean']:+9.4f} "
                  f"{r['wins'] / r['n'] * 100:7.1f}% {r['p']:10.2e} {pa:10.2e}{mark}")
        print()

    print(f"* = Holm-adjusted p < {args.alpha}. A positive mean difference with a "
          f"non-significant p is a tie, and must be reported as one.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ours": our_name, "alpha": args.alpha, "splits": table}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
