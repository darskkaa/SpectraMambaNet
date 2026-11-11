"""Pick how many deep-supervision heads the summed prediction uses, on in-domain only.

Under --ds-mutation the model predicts S1+S2+S3+S_g (net.py). The head count is a published
choice, not a fixed one: Polyp-PVT sums two (Test.py:37, P1+P2), PraNet-V2 sums four
(MyTest_med.py:38). Mutation supervision optimises every summed subset, so 1..4 are all
legally predictable quantities and this is a re-scoring of a finished checkpoint, not a new
model — no training, no parameters, no MACs.

Why it might matter: summing more logits inflates confidence, which costs precision, and our
measured deficit is concentrated on the split where precision decides (CVC-300, -0.021 against
CTNet = 63% of the gap, and below the Polyp-PVT we train ourselves across four runs).

Selection discipline mirrors src/soup.py: k is chosen on Kvasir + CVC-ClinicDB, and ZS-3 is
evaluated exactly once, after k is frozen.

    python src/sweep_heads.py runs/b3max/epoch079.ckpt --report runs/heads_b3max.json

--diagnostic scores every k on every split. That is a test-set read; the resulting grid is for
understanding the model, never for choosing k and never for a reported number.
"""
import argparse
import json

import torch

from test import evaluate_zero_shot
from train import SpectraMambaLightning, ZeroShotMonitorCallback

IN_DOMAIN = ZeroShotMonitorCallback.IN_DOMAIN
ZS3 = ZeroShotMonitorCallback.ZS3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--ema", action="store_true",
                    help="Score the EMA shadow weights instead of the raw ones")
    ap.add_argument("--heads", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="Head counts to try; k means S1..S{k} summed")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--diagnostic", action="store_true",
                    help="Also score every k on the zero-shot splits (test read; not reportable)")
    args = ap.parse_args()

    model = SpectraMambaLightning.load_from_checkpoint(args.checkpoint, map_location="cpu")
    if args.ema:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "ema_state" not in ckpt:
            raise SystemExit(f"{args.checkpoint} carries no ema_state (was --ema-decay set?)")
        model.load_state_dict(ckpt["ema_state"])
    if not getattr(model.model, "ds_mutation", False):
        raise SystemExit("this checkpoint was not trained with --ds-mutation, so only S1 is "
                         "a supervised output and there is nothing to sweep")
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    splits = None if args.diagnostic else IN_DOMAIN
    grid, indomain = {}, {}
    for k in args.heads:
        model.model.ds_infer_heads = k
        res = evaluate_zero_shot(model=model, test_root=args.test_root, splits=splits)
        grid[k] = {s: v["mDice"] for s, v in res.items()}
        indomain[k] = sum(res[s]["mDice"] for s in IN_DOMAIN) / len(IN_DOMAIN)
        print(f"[heads] k={k}  in-domain {indomain[k]:.4f}")

    best = max(indomain, key=indomain.get)
    default = f" vs k=4 ({indomain[4]:.4f})" if 4 in indomain else ""
    print(f"\n[heads] selected k={best} on in-domain ({indomain[best]:.4f}){default}")

    # Selection is closed above; only now may a zero-shot split be read.
    model.model.ds_infer_heads = best
    res = evaluate_zero_shot(model=model, test_root=args.test_root)
    zs3 = sum(res[s]["mDice"] for s in ZS3) / len(ZS3)
    print("\n[heads] final read (selection closed):")
    for s, v in res.items():
        print(f"  {s:24s} {v['mDice']:.4f}")
    print(f"[heads] ZS-3 mDice: {zs3:.4f}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "weights": "ema" if args.ema else "raw",
                "indomain_by_k": indomain,
                "selected_k": best,
                "final_splits": {s: v["mDice"] for s, v in res.items()},
                "final_zs3": zs3,
                "diagnostic_grid": grid if args.diagnostic else None,
            }, f, indent=2)
        print(f"[heads] report {args.report}")


if __name__ == "__main__":
    main()
