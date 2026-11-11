# Cost — conventions, and what was deliberately not shipped

`efficiency.json` holds eight rows measured on one harness: one resolution (352×352), one
batch size (1), one `thop` call, one process. Parameters and MACs do not depend on weights,
so pretrained loading is patched out — several vendored constructors hard-code checkpoint
paths from their original authors' filesystems.

## The convention

Published cost columns in this literature are `thop` **MAC** counts even when the header says
"FLOPs". Two independent anchors confirm our harness sits on that convention:

| model | our harness | that model's own published figure | Δ |
|---|---|---|---|
| Polyp-PVT | 10.02 GMACs | 10.1 | +0.8 % |
| PraNet | 13.15 GMACs | 13.1 | +0.4 % |
| CFA-Net | 55.36 GMACs | 55.36 (its own Table 4) | exact |

Parameter counts run 0.6–2.0 M above published counts on the same models — most likely a
differently counted Res2Net stem. Rankings are unaffected, but the absolute parameter numbers
are ours, not theirs.

## Our own row

```
params        34.854150 M
GMACs traced  16.366
GMACs + floor 18.773
FPS           26.4      (batch 1, 352x352, one NVIDIA L4)
```

`thop` hooks `nn.Module` operations and traces nothing inside the selective-scan kernel, so
every MAC the state-space blocks contribute is invisible to it. **The traced 16.366 is
therefore the lower bound the published convention yields for this model**, and the paper
prints the analytic floor beside it as `16.37 (18.77)` rather than folding it in — folding it
in would break comparability with every other number in that column, all of which are
convolution- and attention-only and therefore complete.

The parameter count is the deployed model. The training-time model is 35.088262 M; the
234,112-parameter difference is the contrastive projection head, which inference discards.
See `MANIFEST.md` §5.

## What was removed rather than shipped

The working repository's `efficiency_measured.json` carried an `ours` record reading
**35.315142 M / 15.679 GMACs**, captioned "SpectraMambaNet at `train.py`'s defaults". It is
wrong for this model twice over: it profiles the argparse defaults rather than the fin2
configuration, and it was taken on CPU, where `mamba_ssm` is unavailable and the scan
degrades to a stub — so the block's real cost is absent from it. The file that produced it
says so in its own footnote.

That record is superseded and is not in this repository. If it turns up in an older document,
it does not describe the model shipped here.

A separate GPU measurement of **39.772 M / 16.277 GMACs** also exists in the working history.
That one is real, but it profiles a different, earlier configuration and likewise does not
describe this model.
