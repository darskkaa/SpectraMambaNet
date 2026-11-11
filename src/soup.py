"""Greedy checkpoint soup, selected on in-domain only (Kvasir + CVC-ClinicDB).

Wortsman-style uniform weight averaging over one run's periodic checkpoints. Selection is
blind to the zero-shot splits by construction: every evaluation inside the greedy loop passes
`splits=IN_DOMAIN` to evaluate_zero_shot, so ZS-3 is computed exactly once, at the end, after
membership is frozen. Averages the raw `state_dict` weights only — never the EMA shadow
(`ema_state` is a separate checkpoint key and is not read), never optimizer state.

    python src/soup.py "lightning_logs/**/epoch*.ckpt" --out soup.ckpt --report soup.json

The glob is `epoch*.ckpt`, not `epoch=*.ckpt`: under --val-fraction 0 the filename template is
"epoch{epoch:03d}", so the files on disk are epoch019.ckpt, epoch039.ckpt, ... An `epoch=*`
pattern matches nothing and this script then exits with "need >=2 checkpoints to soup", which
reads like a missing-checkpoint problem rather than a typo.

The soup is a reportable row only if its in-domain mean beats the best single checkpoint;
otherwise this script says so and the run's normal selected checkpoint stands.
"""
import argparse
import glob
import json
import os

import torch

from test import evaluate_zero_shot
from train import SpectraMambaLightning, ZeroShotMonitorCallback

IN_DOMAIN = ZeroShotMonitorCallback.IN_DOMAIN
ZS3 = ZeroShotMonitorCallback.ZS3


def average_state_dicts(sds):
    """Uniform average of float tensors; non-float entries (step counters etc.) are taken
    from the first member — averaging an int buffer is meaningless."""
    out = {}
    for k, v in sds[0].items():
        if v.is_floating_point():
            acc = sds[0][k].float().clone()
            for sd in sds[1:]:
                acc += sd[k].float()
            out[k] = (acc / len(sds)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def greedy_members(order, solo_scores, score_fn):
    """order: paths sorted by descending solo in-domain score. Returns (members, best_score).
    A candidate stays if the soup's in-domain mean does not drop."""
    members = [order[0]]
    best = solo_scores[order[0]]
    for p in order[1:]:
        cand = score_fn(members + [p])
        if cand >= best:
            members.append(p)
            best = cand
    return members, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+",
                    help="Checkpoint paths or globs (the run's periodic ckpts + last)")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--out", default="soup.ckpt")
    ap.add_argument("--report", default=None, help="Write the full scoring trace as JSON")
    args = ap.parse_args()

    paths = sorted({p for pat in args.checkpoints for p in glob.glob(pat, recursive=True)})
    if len(paths) < 2:
        raise SystemExit(f"need >=2 checkpoints to soup, got {len(paths)}: {paths}")
    print(f"[soup] {len(paths)} ingredients:")
    for p in paths:
        print(f"  {p}")

    model = SpectraMambaLightning.load_from_checkpoint(paths[0], map_location="cpu")
    if torch.cuda.is_available():
        model.cuda()

    sds = {p: torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
           for p in paths}

    def indomain(sd):
        model.load_state_dict(sd)
        model.eval()
        res = evaluate_zero_shot(model=model, test_root=args.test_root, splits=IN_DOMAIN)
        return sum(res[s]["mDice"] for s in IN_DOMAIN) / len(IN_DOMAIN)

    solo = {}
    for p in paths:
        solo[p] = indomain(sds[p])
        print(f"[soup] solo in-domain {solo[p]:.4f}  {os.path.basename(p)}")

    order = sorted(paths, key=lambda p: solo[p], reverse=True)
    members, soup_indom = greedy_members(
        order, solo, lambda ms: indomain(average_state_dicts([sds[m] for m in ms])))

    best_single = order[0]
    print(f"\n[soup] members ({len(members)}/{len(paths)}): "
          f"{[os.path.basename(m) for m in members]}")
    print(f"[soup] in-domain: soup {soup_indom:.4f} vs best single {solo[best_single]:.4f} "
          f"({os.path.basename(best_single)})")
    improved = len(members) > 1 and soup_indom > solo[best_single]
    if not improved:
        print("[soup] NO IMPROVEMENT on in-domain - the soup is not a reportable row; "
              "the run's normally selected checkpoint stands.")

    final_sd = average_state_dicts([sds[m] for m in members])

    # Selection is closed above; only now may a zero-shot split be evaluated.
    model.load_state_dict(final_sd)
    model.eval()
    print("\n[soup] final read (selection closed - first and only ZS-3 evaluation):")
    res = evaluate_zero_shot(model=model, test_root=args.test_root)
    zs3 = sum(res[s]["mDice"] for s in ZS3) / len(ZS3)
    print(f"[soup] ZS-3 mDice: {zs3:.4f}")

    ckpt = torch.load(members[0], map_location="cpu", weights_only=False)
    for k in ("optimizer_states", "lr_schedulers", "ema_state", "callbacks", "loops"):
        ckpt.pop(k, None)
    ckpt["state_dict"] = final_sd
    ckpt["soup_members"] = members
    torch.save(ckpt, args.out)
    print(f"[soup] saved {args.out}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({
                "ingredients": paths,
                "solo_indomain": solo,
                "members": members,
                "soup_indomain": soup_indom,
                "best_single": best_single,
                "best_single_indomain": solo[best_single],
                "improved_indomain": improved,
                "final_splits": {k: v["mDice"] for k, v in res.items()},
                "final_zs3": zs3,
            }, f, indent=2)
        print(f"[soup] report {args.report}")


if __name__ == "__main__":
    main()
