"""Dump the per-epoch scalars a training run wrote to TensorBoard.

train.py logs `tel/ls_h`, `tel/ls_w`, `tel/ls_f`, `tel/edge_mean`, `tel/band_frac`,
`tel/abs_delta`, `tel/rfp_*` to the Lightning logger only — none of them reach the
zero-shot history JSON, so they have never been read. They answer, with no GPU:
whether the Mamba's LayerScales ever left their 1e-4 init, whether the edge map
collapsed toward 0 (CCM degenerating to `s`), and whether BoundaryRegionCalib moved
off its zero-init identity.

    python src/analysis/read_telemetry.py <lightning_logs/version_N or a parent dir>
"""
import argparse
import csv
import glob
import os
import sys


def find_event_dirs(root):
    if glob.glob(os.path.join(root, "events.out.tfevents.*")):
        return [root]
    hits = glob.glob(os.path.join(root, "**", "events.out.tfevents.*"), recursive=True)
    return sorted({os.path.dirname(h) for h in hits})


def read_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return {t: ea.Scalars(t) for t in ea.Tags()["scalars"]}


def step_to_epoch(scalars):
    """Map global step -> epoch using the `epoch` scalar Lightning always logs."""
    if "epoch" not in scalars:
        return None
    marks = sorted((e.step, int(e.value)) for e in scalars["epoch"])

    def lookup(step):
        best = marks[0][1]
        for s, ep in marks:
            if s > step:
                break
            best = ep
        return best
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="lightning_logs/version_N, or any dir above it")
    ap.add_argument("--csv", default=None, help="write the full table here")
    ap.add_argument("--tags", default=None,
                    help="comma-separated substring filter (default: every tag)")
    args = ap.parse_args()

    dirs = find_event_dirs(args.run_dir)
    if not dirs:
        sys.exit(f"no events.out.tfevents.* under {args.run_dir}")

    for run_dir in dirs:
        scalars = read_scalars(run_dir)
        if args.tags:
            wanted = [w.strip() for w in args.tags.split(",")]
            scalars = {t: v for t, v in scalars.items()
                       if t == "epoch" or any(w in t for w in wanted)}
        print(f"\n=== {run_dir}")
        if not scalars:
            print("  no scalars")
            continue

        tel = sorted(t for t in scalars if t.startswith("tel/"))
        if not tel:
            print("  NOTE: no tel/* tags. Either this run predates the telemetry logging")
            print("  (train.py:451-472) or it is a --model polyp_pvt run, which has no")
            print("  Mamba LayerScales, no edge head and no BoundaryRegionCalib to report.")

        print(f"  {'tag':<24} {'n':>5} {'first':>12} {'min':>12} {'max':>12} {'last':>12}")
        print("  " + "-" * 82)
        for tag in sorted(scalars):
            if tag == "epoch":
                continue
            vals = [e.value for e in scalars[tag]]
            print(f"  {tag:<24} {len(vals):>5} {vals[0]:>12.6g} {min(vals):>12.6g} "
                  f"{max(vals):>12.6g} {vals[-1]:>12.6g}")

        to_epoch = step_to_epoch(scalars)
        if tel and to_epoch:
            print(f"\n  per-epoch tel/* ({run_dir})")
            print("  epoch  " + "  ".join(f"{t[4:]:>12}" for t in tel))
            rows = {}
            for tag in tel:
                for e in scalars[tag]:
                    rows.setdefault(to_epoch(e.step), {})[tag] = e.value
            for ep in sorted(rows):
                cells = "  ".join(f"{rows[ep].get(t, float('nan')):>12.6g}" for t in tel)
                print(f"  {ep:>5}  {cells}")

        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
            tags = [t for t in sorted(scalars) if t != "epoch"]
            rows = {}
            for tag in tags:
                for e in scalars[tag]:
                    key = (to_epoch(e.step) if to_epoch else None, e.step)
                    rows.setdefault(key, {})[tag] = e.value
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "step"] + tags)
                for (ep, step) in sorted(rows, key=lambda k: k[1]):
                    r = rows[(ep, step)]
                    w.writerow([ep, step] + [r.get(t, "") for t in tags])
            print(f"\n  wrote {args.csv}")


if __name__ == "__main__":
    main()
