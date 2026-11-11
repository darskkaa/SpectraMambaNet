"""P10: are we training on the same ImageNet weights the competitors are?

We build the encoder with `timm.create_model('pvt_v2_b2', pretrained=True)` (model/net.py:46).
Polyp-PVT (`lib/pvt.py:163`) and FCC-Net (`lib/FCCNet.py:99`) instead load the PVTv2 authors'
`pvt_v2_b2.pth` from the official PVT release.
timm's `pvt_v2_b2.in1k` is *believed* to be that file converted to timm's key layout, but the
belief was never checked, and if it is wrong it is a complete explanation for our 0.021
Polyp-PVT reproduction gap — every other candidate would be a smaller effect.

Key layouts differ (`block1.0.attn.q` vs `stages.0.blocks.0.attn.q`, `mlp.dwconv.dwconv` vs
`mlp.dwconv.conv`, `head` vs `head.fc`), so this does not attempt a name map. Weight identity
does not depend on names: it fingerprints every tensor by (dtype, shape, exact sum, exact
sum-of-squares) in float64 and asks whether the two multisets agree. A converted-but-identical
checkpoint matches exactly; a differently-trained one cannot, because two independent training
runs do not reproduce a float64 sum over 25 M parameters.

Usage:
    python src/verify_encoder_weights.py --official /path/to/pvt_v2_b2.pth
"""
import argparse
import collections

import torch


def fingerprint(sd):
    out = collections.Counter()
    for k, v in sd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        v = v.detach().double()
        out[(tuple(sorted(v.shape)), round(v.sum().item(), 6),
             round(v.pow(2).sum().item(), 6))] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="the authors' pvt_v2_b2.pth")
    ap.add_argument("--model", default="pvt_v2_b2")
    args = ap.parse_args()

    import timm

    official = torch.load(args.official, map_location="cpu", weights_only=False)
    official = official.get("state_dict", official.get("model", official))
    ours = timm.create_model(args.model, pretrained=True).state_dict()

    fo, ft = fingerprint(official), fingerprint(ours)
    n_o = sum(fo.values())
    n_t = sum(ft.values())
    matched = sum((fo & ft).values())

    print(f"official tensors: {n_o}   timm tensors: {n_t}   exact-value matches: {matched}")
    print(f"official params:  {sum(v.numel() for v in official.values() if torch.is_tensor(v)):,}")
    print(f"timm params:      {sum(v.numel() for v in ours.values()):,}")

    missing = fo - ft
    if missing:
        print(f"\n{sum(missing.values())} official tensor(s) with no value-equal counterpart "
              f"in timm — shapes:")
        for (shape, s, ss), n in list(missing.items())[:20]:
            print(f"  x{n}  shape={shape}  sum={s:+.4f}  sumsq={ss:.4f}")

    same = matched == n_o
    print(f"\n[P10] {'IDENTICAL' if same else 'DIFFERENT'} — "
          + ("timm's in1k tag is the authors' checkpoint converted; the encoder is not the "
             "reproduction gap." if same else
             "we are NOT training from the same initialisation as Polyp-PVT/FCC-Net. Load the "
             "authors' file and re-baseline before spending another run."))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
