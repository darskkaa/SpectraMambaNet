# Run analysis — what `logs/run_final_2.txt` actually shows

Read alongside `MANIFEST.md`, which pins the run. This file is about behaviour: how the run
progressed, where it peaked, what the telemetry says, and which parts of the log are easy to
misread.

Everything below was re-derived from the log; `results/fin2_eval_trajectory.json` carries the
parsed form.

---

## 1. Shape of the run

| | |
|---|---|
| Structure | 1114 lines: a 13.2 min pre-flight audit, then 160 epochs |
| Duration | 8.42 h, `exit 0`, `max_epochs=160 reached` |
| Interruptions | none — one continuous session. The second `[1/10] Environment Check` block at `:381` is the launcher re-running its audit after the dry run, not a resume |
| Evaluations | 32, every 5 epochs from epoch 5 to epoch 160 |
| Checkpoints written | 5 `best-on-test.ckpt` saves, plus 8 cycle-end snapshots |

Training loss falls from **6.703** at epoch 0 to **1.0788** at epoch 157, ending at 1.0846.
It is monotone in trend but visibly sawtoothed — each cosine restart at a multiple of 20
lifts the loss before it settles lower, which is the schedule working as intended rather than
instability.

`Val Dice: N/A` on every epoch line is expected: `--val-fraction 0` means there is no
held-out validation split. All 1450 training pairs are used for training, following the
PraNet convention.

---

## 2. Where the run actually peaked

Accuracy does not improve monotonically with epochs, and the last epoch is not the best one.

| statistic | best epoch | value |
|---|---|---|
| in-domain mean (Kvasir + ClinicDB) | 150 | 0.9326 |
| five-set mean | 80 | 0.8753 |
| Kvasir | 125 | 0.9289 |
| CVC-ClinicDB | 150 | 0.9404 |
| CVC-300 | 70 | 0.9123 |
| CVC-ColonDB | 120 | 0.8274 |
| ETIS-LaribPolypDB | 95 | 0.8266 |

**Every split peaks at a different epoch, and none of them peaks at 160.** Epoch 150 is the
best the model ever gets at home and one of its weaker showings away. This is why the
reported model is a soup rather than a single epoch, and why the selection rule is restricted
to cycle ends — see `MANIFEST.md` §6 for the full argument and the arithmetic that rules out
test selection.

The unseen centres also wander between adjacent evaluations with no trend:

| split | mean adjacent \|Δ\| | max |
|---|---|---|
| CVC-300 | 0.0113 | 0.0446 |
| CVC-ColonDB | 0.0101 | 0.0351 |
| ETIS-LaribPolypDB | 0.0195 | 0.0590 |

Single-evaluation differences of that size on those splits are not distinguishable from the
wander, and nothing in this repository treats them as if they were.

---

## 3. Telemetry

Every epoch line is followed by a `[tel]` block. The channels worth knowing:

| channel | what it measures |
|---|---|
| `h4_ls_*`, `h8_ls_*` | LayerScale gates on the height, width and third branches of each block |
| `h4_delta_mag`, `h8_delta_mag` | mean absolute pre-bound log-gain on the third branch |
| `abs_delta`, `band_frac` | magnitude and spatial extent of the boundary calibration |
| `edge_mean` | edge-head activation |
| `proto_ctx` | region-prototype context weight |

Measured over all 160 epochs:

| channel | epoch 1 | max | epoch 160 |
|---|---|---|---|
| `h4_delta_mag` | 0.0027 | 1.2920 | 0.3817 |
| `h8_delta_mag` | 0.0007 | 0.8917 | 0.5540 |
| `h4_ls_f` | 0.0012 | 0.0508 | 0.0508 |
| `h8_ls_f` | 0.0005 | 0.0732 | 0.0732 |
| `band_frac` | 0.0743 | 0.0743 | 0.0586 |
| `abs_delta` | 0.0006 | 0.1086 | 0.0529 |
| `proto_ctx` | 0.0136 | 1.0480 | 0.9254 |

Three readings matter.

**The block is active, not inert.** Both `delta_mag` channels start within rounding of zero —
the branch is an exact identity at initialisation by construction — and end two to three
orders of magnitude higher. A branch that never left its zero initialisation would still
report 0 at epoch 160. The LayerScale gates tell the same story: initialised at `1e-4`, they
climb to 0.05 and 0.07 and hold there for the rest of the run.

**Nothing was gradient-dead.** The gain is bounded at 3.0. The largest pre-bound value seen
anywhere in the run is 1.292, and both channels settle well under that. The branch never ran
against its range — which is the failure mode that killed an earlier arm and prompted the
bounded redesign.

**The calibration head stays band-confined.** `band_frac` averages 0.0597 across the second
half of the run and never exceeds 0.0670: the correction reaches about 6% of pixels rather
than diffusing across the image. That is the property the design intends, and this is the
channel that would expose it if it stopped holding.

---

## 4. Three ways to misread this log

**`[params] total=35.088 M` is not the reported parameter count.** The training model carries
a 234,112-parameter contrastive projection head that inference discards. The deployed model
is 34,854,150 parameters. `scripts/verify_release.py` constructs both and asserts the
subtraction.

**`--select-on-test` does not mean the test sets chose the model.** The flag enables the
published max-over-epochs convention so the number is comparable to how competitors report
theirs. The statistic being maximised is set by `--select-on indomain`, and the log's own
save lines say so: `selected on raw indomain 0.9326 (ZS-3 stays blind)`. `MANIFEST.md` §6
shows what a test-reading rule would have picked instead, and it is a different epoch.

**The five `best-on-test.ckpt` saves are not the shipped weights.** That file tracks the
running in-domain best and last moved at epoch 150. The shipped weights are a soup of epochs
80 and 120, built afterwards by `src/soup.py`; the checkpoint's `soup_members` field names
both.

---

## 5. Cost

Measured separately, not in this run — see `results/efficiency.json`, and `results/COST.md`
for the conventions and the one stale measurement that was removed from this repository
rather than shipped.

At 352×352 and batch 1 the deployed model is **34.854150 M parameters** and **16.366 GMACs**
traced, **18.773** with an analytic floor added for the scans `thop` cannot see, running at
26.4 FPS on one L4. The parameter count is the deployed one, matching §4 and the paper — not
the 35.088 M the training run printed. The paper's cost table prints the traced 16.37 alone;
`MANIFEST.md` §5 explains why `results/COST.md` still shows `16.37 (18.77)`.
