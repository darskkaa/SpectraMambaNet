"""The gate that makes `model/mamba_ref.py` publishable.

The pure-PyTorch scan is only trustworthy if a checkpoint trained with the real CUDA
kernel loads into it and reproduces the scores that checkpoint already has. Two stages:

  1. **Key parity** — every `mamba_*` tensor in the checkpoint must match a freshly built
     block by name AND shape, with nothing missing and nothing extra. This catches a
     wrong `d_state`/`dt_rank`/`expand` immediately and needs no GPU time.
  2. **Score parity** — run the canonical evaluator and compare per-split mDice against
     the numbers recorded for that checkpoint. Agreement to --tol proves the math.

The shipped weights' recorded scores (results/fin2_soup8_scored.json) are the defaults, so:

    python scripts/verify_mamba_ref.py fin2_soup8.ckpt \
        --test-root TestDataset/TestDataset

Stage 1 alone runs on any real-kernel checkpoint.
A FAIL here means every map and figure produced locally is suspect — use Colab instead.
"""
import argparse
import json
import os
import re
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

with open(os.path.join(REPO, "results", "fin2_soup8_scored.json")) as f:
    EXPECT = {s: r["mDice"] for s, r in json.load(f)["splits"].items() if isinstance(r, dict)}
ZS3 = ["CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB"]


def stage1(ckpt_path):
    from model.net import TriAxialMambaBottleneck
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)

    # Group the checkpoint's Mamba tensors by their owning module. Match only the block's own
    # scan attributes: the stages holding the blocks are themselves named mamba_h4/mamba_h8.
    blocks = {}
    for k, v in sd.items():
        m = re.fullmatch(r"(.+)\.mamba_([hwf](?:_rev)?)\.(.+)", k)
        if m:
            head, axis, param = m.groups()
            blocks.setdefault((head, axis), {})[param] = tuple(v.shape)
    if not blocks:
        print("  no mamba_* tensors in this checkpoint — was it trained on the stub?")
        return False

    ok = True
    seen_dims = set()
    for (head, axis), params in sorted(blocks.items()):
        if "in_proj.weight" not in params:
            print(f"  {head}.mamba_{axis} has no in_proj — this checkpoint was trained on "
                  f"the old Linear+GELU stub ({', '.join(sorted(params))}), not a real "
                  f"Mamba. Nothing to verify against.")
            return False
        d_model = params["in_proj.weight"][1]
        seen_dims.add(d_model)
        ref = {n: tuple(p.shape) for n, p in
               TriAxialMambaBottleneck(d_model).mamba_h.state_dict().items()}
        missing = sorted(set(ref) - set(params))
        extra = sorted(set(params) - set(ref))
        wrong = sorted(n for n in set(ref) & set(params) if ref[n] != params[n])
        if missing or extra or wrong:
            ok = False
            print(f"  MISMATCH {head}.mamba_{axis} (d_model={d_model})")
            for n in missing:
                print(f"    missing  {n} {ref[n]}")
            for n in extra:
                print(f"    extra    {n} {params[n]}")
            for n in wrong:
                print(f"    shape    {n} ckpt {params[n]} != ref {ref[n]}")
    print(f"  {len(blocks)} Mamba modules at d_model {sorted(seen_dims)}: "
          f"{'ALL KEYS AND SHAPES MATCH' if ok else 'MISMATCH'}")
    return ok


def stage2(ckpt_path, test_root, tol):
    from test import evaluate_zero_shot
    results = evaluate_zero_shot(checkpoint_path=ckpt_path, test_root=test_root)
    got = {s: r["mDice"] for s, r in results.items() if isinstance(r, dict) and "mDice" in r}
    ok = True
    for split, want in EXPECT.items():
        have = got.get(split)
        if have is None:
            print(f"  {split:20s}     ---     (not scored)")
            ok = False
            continue
        bad = abs(have - want) > tol
        ok &= not bad
        print(f"  {split:20s} {have:.4f}  expect {want:.4f}  "
              f"delta {have - want:+.4f}{'   <-- FAIL' if bad else ''}")
    if all(s in got for s in ZS3):
        z, zw = (sum(d[s] for s in ZS3) / 3 for d in (got, EXPECT))
        print(f"  {'ZS-3':20s} {z:.4f}  expect {zw:.4f}  delta {z - zw:+.4f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--test-root", default=os.path.join(REPO, "TestDataset", "TestDataset"))
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--keys-only", action="store_true",
                    help="stage 1 only; no GPU pass over the 798 test images")
    args = ap.parse_args()

    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(REPO, args.checkpoint)
    if not os.path.isfile(ckpt):
        sys.exit(f"no such checkpoint: {ckpt}")

    print(f"device: {'cuda — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    try:
        import mamba_ssm  # noqa: F401
        print("NOTE: the real mamba_ssm kernel is installed, so this run does not "
              "exercise mamba_ref at all.")
    except ImportError:
        pass

    print("\n[1/2] state-dict key and shape parity")
    ok1 = stage1(ckpt)
    if args.keys_only:
        sys.exit(0 if ok1 else 1)

    print("\n[2/2] score parity on the canonical evaluator")
    ok2 = stage2(ckpt, args.test_root, args.tol)

    print("\n" + ("PASS — maps and figures produced locally are trustworthy."
                  if ok1 and ok2 else
                  "FAIL — do not publish anything produced through mamba_ref; use Colab."))
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
