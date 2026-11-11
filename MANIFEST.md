# Manifest — exact state of run 2 (`fin2`)

Everything in this repository descends from one training run. This file pins that run to an
exact commit, an exact command, an exact environment and an exact set of weights, and shows
the arithmetic connecting each to the next. Nothing here is copied from another write-up;
every value was re-derived from `logs/run_final_2.txt` or from the checkpoint itself.

Compiled 2026-08-14; updated 2026-11-02.

---

## 1. What run 2 is

| | |
|---|---|
| Run name | `fin2` |
| Model | SpectraMambaNet — PVTv2-B2 encoder, two tri-axial state-space blocks on `f1`/`f0`, FCC octave-convolution decoder, zero-init band-confined boundary calibration |
| Schedule | 160 epochs, cosine restarts, cycle length 20, 5 warmup epochs |
| Wall clock | **8.42 h** training + 13.2 min pre-flight (`logs/run_final_2.txt`, last line and `:379`) |
| Exit | `exit 0` — `Trainer.fit stopped: max_epochs=160 reached`. The run completed; it was not truncated or resumed |
| Shipped weights | greedy weight-space soup of **2 of the 8** cycle-end snapshots |

---

## 2. Exact code state

### The commit that trained it

```
65b63c54d0a2d777a757a694cd78574bf370a113
2026-08-10 02:14:46 -0400
"Park the unsuffixed backup folder and clear stale logs on a fresh fin1 start"
repo darskkaa/polp_wacv (private), branch sota-fcc-boundary-calib
```

**How this was established.** The launcher clones the branch head at launch time
(`logs/run_final_2.txt:280`, `Cloning … (sota-fcc-boundary-calib)`) and the training session
reports `Already up to date`, so no commit landed between pre-flight and launch. The log
carries no SHA, so the commit is pinned by time and by content:

- The log file was written out at 2026-08-10 22:45 local. Subtracting the reported
  8.42 h of training and 0.22 h of pre-flight places launch at roughly **14:00 on 2026-08-10**.
- The branch history has **no commit at all** between `65b63c5` (02:14) and `0b32fa9` (14:45),
  so `65b63c5` was the branch head across the entire launch window.
- `0b32fa9` and `a377a46` (14:45, 14:50) touch only `docs/`, `scripts/make_ablation_table.py`
  and `src/profile_efficiency.py` — none of which training imports. So even if the clone
  landed on the far side of them, the trained code is identical.

### What changed afterwards, and why it does not matter

`src/model/net.py` and `src/train.py` were both modified once after this run, by
`39c8a342198fee90116d7e58ff7429cff19eab16` (2026-08-10 21:14). That commit added four
switches — `--freq-gain`, `--freq-gain-scale`, `--freq-gain-shared`, `--third-axis` — plus a
pure-PyTorch fallback scan and one extra telemetry channel.

**All four switches default to the behaviour they replaced**, verified against
`src/train.py:1036-1059` and `src/model/net.py:319-330`:

| flag | default | pre-`39c8a34` behaviour |
|---|---|---|
| `--freq-gain` | `clamp` | `torch.clamp(delta_mag, min=-3.0, max=3.0)` — identical |
| `--freq-gain-scale` | `3.0` | hardcoded `3.0` — identical |
| `--freq-gain-shared` | off | `mag_head = nn.Linear(dim, dim)` — identical shape |
| `--third-axis` | `freq` | the spectral branch was the only branch — identical |

The only other addition is `last_delta_sat`, a detached scalar logged to TensorBoard. It
enters no loss and no parameter.

**The consequence, checked rather than asserted:** the code in `src/` loads the shipped
weights with `strict=True` and reports the same parameter count the run printed. Re-run
`scripts/verify_release.py` to reproduce that check.

### Changes made for this public release

The development repository is private, so `65b63c5` itself is not published. This repository
is that tree plus `39c8a34` (above) plus the edits below. With their defaults, none of them
changes what a training run reads or computes.

| file | change | effect on a training run |
|---|---|---|
| `src/train.py` | new `--train-root` and `--test-root`, defaulting to `$TRAIN_DATA_ROOT` and `$TEST_DATA_ROOT`, then to `/content/TrainDataset` and `/content/TestDataset` | none at the defaults, which are the paths the run read (`logs/run_final_2.txt:405-425`) |
| `src/test.py` | fetches PraNet's public `TestDataset.zip` when the test root is missing (`TESTSET_DRIVE_ID=""` opts out); loads checkpoints with `pretrained_encoder=False` | none. Training evaluates the in-memory model, and a checkpoint overwrites every encoder weight the skipped ImageNet download would have set |
| `src/data/datamodule.py` | removed an unused `from datasets import load_dataset` | none |
| `colab_train_launcher.py` | clones this repository's `main`; downloads PraNet's public train and test zips unless `POLYP_DATA_FOLDER_ID` names a Drive folder; pins `causal-conv1d==1.6.2.post1` and `mamba-ssm==2.3.2.post1` | none. The pins are the wheels the run built (`logs/run_final_2.txt:307-308`) |
| `src/paired_significance.py` | `--others` defaults to `results/competitors/`, which now ships with the repository | evaluation only |
| `requirements.txt`, `requirements_colab.txt` | list what `src/` imports; the CUDA kernels moved to the second file | install only |
| `LICENSE`, `NOTICE` | new: MIT for our code; `NOTICE` lists the third-party code, weights and data with their licences or terms | none |

Also added for this release: `configs/ablation_commands.sh`, `logs/ablation_R0-R3.txt`, the
eleven competitor score files in `results/competitors/`, and sections G and H of
`scripts/verify_release.py`, which re-derive the ablation and significance numbers from them.

`logs/run_final_2.txt` names the development repository at lines 14 and 280, where the review
copy printed `ANONYMIZED`. Those two lines are the only edits to it, and
`logs/ablation_R0-R3.txt` is the sweep's console output with no edits. Git stores both with
LF line endings.

---

## 3. Exact environment

Read from `logs/run_final_2.txt:265` and `:386` — printed twice, identically, once in
pre-flight and once at launch.

```
Python  : 3.12.13
PyTorch : 2.11.0+cu128
CUDA    : 12.8
GPU     : NVIDIA L4
Lightning: 2.6.5          (from the checkpoint's pytorch-lightning_version field)
Precision: bf16-mixed, float32_matmul_precision=medium
mamba-ssm: 2.3.2.post1    (wheel names at :307-308)
causal-conv1d: 1.6.2.post1
```

Datasets, audited at launch (`:405-425`) and matching the counts in `predictions/` exactly:

| split | images | masks | role |
|---|---|---|---|
| Kvasir | 100 | 100 | in-domain test |
| CVC-ClinicDB | 62 | 62 | in-domain test |
| CVC-ColonDB | 380 | 380 | unseen centre |
| ETIS-LaribPolypDB | 196 | 196 | unseen centre |
| CVC-300 | 60 | 60 | unseen centre |
| TrainDataset | 1450 | 1450 | full set, no val holdout (PraNet convention) |

---

## 4. Exact command

Reproduced verbatim from `logs/run_final_2.txt:437`, and kept in
`configs/fin2_train_command.sh`:

```
python -u src/train.py --fresh --model ours --encoder pvt_v2_b2 --epochs 160 --seed 0 \
  --precision bf16-mixed --lr-schedule cosine-restarts --restart-cycle-len 20 \
  --warmup-epochs 5 --bce-unweighted --iou-smooth 1.0 --ms-mask-interp bilinear \
  --boundary-dice-weight 0.3 --contrast-weight 0.1 --fcc-channel 64 --fcc-region-proto \
  --edge-stride 4 --mamba-stage h4h8 --mamba-scan raster --mamba-bidirectional \
  --ds-mutation --aug endo --hat-prob 0.5 --val-fraction 0 --select-on indomain \
  --select-weights raw --select-on-test --zeroshot-every 5 \
  --zeroshot-history runs/fin2_seed0.json
```

The 51 settings this resolves to are in `configs/fin2_run_config.json`, parsed directly from
the `[run-config]` line the run printed at `:459`.

Two settings are worth reading off explicitly because they are easy to assume wrong:

- `ema_decay=0.0` — **this run used no EMA.** Every number is raw weights.
- `val_fraction=0.0` — there is no held-out validation split. Selection ran on the two
  in-domain test sets; see §6.

---

## 5. The weights

Not in git — 140 MB exceeds GitHub's blob limit. Published as a release asset; see the README.

| | |
|---|---|
| File | `fin2_soup8.ckpt` |
| Size | 140,669,680 bytes |
| SHA-256 | `598de21496dbc7f7c0619ea0d2401c7dc28b6d5885c638359d695e0c69401e0b` |
| Format | Lightning checkpoint, 863 tensors under `state_dict` |
| `epoch` | 79 (0-indexed → epoch 80) |
| `global_step` | 21600 |
| `soup_members` | `epochepoch=079.ckpt`, `epochepoch=119.ckpt` → **epochs 80 and 120** |

`hyper_parameters` inside the checkpoint carries 46 of the 51 config keys and agrees with
`configs/fin2_run_config.json` on every one they share.

### Parameter count — the 35.088 / 34.854 reconciliation

Two different numbers are correct, for two different models:

| model | parameters | where it appears |
|---|---|---|
| as trained, with the contrastive projection head | **35,088,262** = 35.088 M | `logs/run_final_2.txt:464` |
| as deployed and as profiled, head discarded | **34,854,150** = 34.854150 M | the paper's cost table |

The difference is exactly the projection head, `35,088,262 − 34,854,150 = 234,112`, which
`--contrast-weight 0.1` attaches during training and which contributes nothing at inference.
Breakdown of the trained model:

```
encoder        24,849,856
mamba_h8          749,568
mamba_h4          208,896
fcc_decoder     9,045,830
contrast_head     234,112      <- train-only
               ----------
total          35,088,262
```

The checkpoint keeps the head, so it loads `strict=True` against the training-time
construction and reports unexpected keys against the inference-time one. Both are expected;
`scripts/verify_release.py` asserts both directions.

### Compute

`results/COST.md` and the `ours` record in `results/efficiency.json` give two MAC counts at
352×352: **16.366 G** traced by `thop`, and **18.773 G** with an analytic floor added for the
selective scans `thop` cannot trace. Since 2026-08-15 the paper's cost table prints the traced
16.37 alone. Both files still describe the paper as printing `16.37 (18.77)`, which was the
draft's wording; the two values themselves are unchanged.

---

## 6. How the shipped weights were selected

This is the step most worth checking, because the selection rule and the reported score can
look circular. They are not, and the log proves it.

Eight snapshots were taken, one at each cosine-restart cycle end. Each was scored on the two
in-domain sets, and the selection statistic is their mean. Those eight means are **not
printed anywhere in the log** — they are recomputed here from the per-dataset Dice values in
the eight matching evaluation blocks:

| cycle end | in-domain (selection) | five-set mean |
|---|---|---|
| epoch 20 | 0.9234 | 0.8679 |
| epoch 40 | 0.9224 | 0.8748 |
| epoch 60 | 0.9211 | 0.8675 |
| **epoch 80** | **0.9242** | 0.8753 |
| epoch 100 | 0.9158 | 0.8724 |
| epoch 120 | 0.9228 | 0.8728 |
| epoch 140 | 0.9132 | 0.8663 |
| epoch 160 | 0.9196 | 0.8701 |

Epoch 80 is the pool maximum on the selection statistic, and it is also where the five-set
mean happens to peak. That coincidence is the thing to be careful about, so here is the
disproof of test selection: across **all 32** evaluation points the in-domain maximum is
**epoch 150 at 0.9326**, whose five-set mean is only **0.8642**. A rule that read the test
sets would have taken epoch 150. The rule that was actually applied — pool restricted to
cycle ends, ranked on in-domain only — takes epoch 80.

`results/fin2_eval_trajectory.json` holds all 32 points so this can be re-checked directly.

---

## 7. Results carried by this repository

Soup of epochs 80 + 120, scored on the maps in `predictions/`:

| dataset | mDice | mIoU | Recall@0.5 | HD95@0.5 | images |
|---|---|---|---|---|---|
| Kvasir | 0.9274 | 0.8838 | 0.9193 | 36.56 px | 100 |
| CVC-ClinicDB | 0.9386 | 0.8935 | 0.9584 | 6.81 px | 62 |
| CVC-300 | 0.8980 | 0.8326 | 0.9589 | 13.37 px | 60 |
| CVC-ColonDB | 0.8308 | 0.7531 | 0.8670 | 39.30 px | 380 |
| ETIS-LaribPolypDB | 0.8219 | 0.7446 | 0.9202 | 102.95 px | 196 |

Mean over the five splits: **0.883335**.

Full per-image scores for all 798 images are in `results/fin2_soup8_scored.json` under each
split's `per_image` key. Paired significance results are in `results/significance_fin2.json`.

---

## 8. What is deliberately absent

| absent | why |
|---|---|
| Training and test image data | redistribution is the dataset owners' call. The code reads PraNet's public archives: [TrainDataset.zip](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view) (1,450 pairs) and [TestDataset.zip](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view) (the five splits) |
| The eight per-cycle snapshots | ~5 GB. Only the soup is needed to reproduce every number here |
| The 200-epoch legacy run | a different architecture. Its numbers do not describe this model and are not carried forward |
| Console logs of the third-axis arms (M3/M5) | they ran in a separate session whose output was not kept, so their Dice values cannot be re-derived here. Their commands are rebuilt in `configs/ablation_commands.sh` and their parameter counts are checked. The R0-R3 sweep log is `logs/ablation_R0-R3.txt` |

---

## 9. Re-verification

```bash
python scripts/verify_release.py            # needs fin2_soup8.ckpt in the repo root
python scripts/verify_release.py --no-ckpt  # everything except the weight checks
```

129 checks with the weights present, 119 without. The script re-derives, from the files in
this repository and not from this document: the parameter counts and their reconciliation,
the strict load in both directions, the checkpoint hash and metadata, the eight cycle-end
selection scores, the pool argmax, the global argmax that disproves test selection, every
cell of the results table, the five-set mean, the prediction-map counts, every R0-R3 ablation
cell, mean, step and cost from `logs/ablation_R0-R3.txt`, the parameter count of every
ablation arm, and all 396 values of the paired-significance table, re-run from the per-image
scores.

Inference was also re-run end to end from the release checkpoint on an RTX 2060 SUPER under
Windows, with `src/model/mamba_ref.py` in place of the CUDA scan. Every mean Dice, mean IoU
and Recall@0.5 in §7 reproduced to four decimals, per-image Dice moved by at most 0.0008,
and split HD95 by at most 0.02 px.
