"""Inference-time branch knockout on a trained checkpoint.

`TriAxialMambaBottleneck.forward` ends at `out_w_spatial + self.ls_f(fused)` (net.py:336),
and the bi-axial path returns `out_w_spatial` alone (net.py:275-276). So zeroing every
`ls_f.gamma` makes the block return *exactly* what `--no-mamba-freq-axis` would return —
the knockout is identity with the ablated architecture, not an approximation of it. The
same holds for ls_h / ls_w, which gate the two spatial scans.

    python src/knockout_axis.py runs/fin2_soup8.ckpt --test-root /content/TestDataset

This measures whether a branch is load-bearing *in the shipped weights*. It is not a
substitute for a retrained control: the rest of the network trained in the branch's
presence and cannot re-allocate here, so the drop is an upper bound on what retraining
without the branch would cost. It is, however, exactly the test that found FARF-Net's ARF
and HGM's three modules inert, so a paper that runs it on competitors has to run it on
itself.
"""
import argparse
import json

import torch

from test import evaluate_zero_shot
from train import SpectraMambaLightning, ZeroShotMonitorCallback

SPLITS = ["Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB"]
ZS3 = ZeroShotMonitorCallback.ZS3


def zeroed(sd, spec):
    out = {k: v.clone() for k, v in sd.items()}
    hits = []
    for axis in spec.split("+"):
        suffix = f"ls_{axis}.gamma"
        found = [k for k in sd if k.endswith(suffix)]
        if not found:
            raise SystemExit(f"no parameter ends in '{suffix}' — checkpoint has no {axis} branch")
        for k in found:
            out[k].zero_()
        hits += found
    return out, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--axes", default="f,h,w",
                    help="Comma-separated arms; join with '+' to knock out several at once "
                         "(e.g. 'f,h,w,f+h+w' adds a whole-block-off row)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    model = SpectraMambaLightning.load_from_checkpoint(args.checkpoint, map_location="cpu")
    if torch.cuda.is_available():
        model.cuda()
    base_sd = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)["state_dict"]

    def score(sd, label):
        model.load_state_dict(sd)
        model.eval()
        print(f"\n=== {label} ===")
        res = evaluate_zero_shot(model=model, test_root=args.test_root)
        return {k: v["mDice"] for k, v in res.items()}

    # A LayerScale the network has grown large but whose removal costs nothing is the
    # signature the audit looks for in competitors; print it for ourselves too.
    print("\nLayerScale gamma L1-norm per gate (what the network chose to weight):")
    for k, v in sorted(base_sd.items()):
        if k.endswith(".gamma"):
            print(f"  {k:<60} {v.abs().mean().item():.5f}")

    rows = {"intact": score(base_sd, "intact")}
    for axis in [a.strip() for a in args.axes.split(",") if a.strip()]:
        sd, hits = zeroed(base_sd, axis)
        rows[f"ls_{axis}=0"] = score(sd, f"ls_{axis} zeroed ({len(hits)} tensors)")

    def zs3(r):
        keys = {k.upper(): k for k in r}
        return sum(r[keys[s.upper()]] for s in ZS3) / len(ZS3)

    print("\n" + "=" * 78)
    header = f"{'arm':<12}" + "".join(f"{s[:11]:>13}" for s in SPLITS) + f"{'ZS-3':>9}"
    print(header)
    print("-" * len(header))
    for label, r in rows.items():
        keys = {k.upper(): k for k in r}
        cells = "".join(f"{r[keys[s.upper()]]:>13.4f}" for s in SPLITS)
        print(f"{label:<12}{cells}{zs3(r):>9.4f}")
    print("-" * len(header))
    intact = zs3(rows["intact"])
    for label, r in rows.items():
        if label != "intact":
            print(f"  {label:<12} ZS-3 delta vs intact: {zs3(r) - intact:+.4f}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
