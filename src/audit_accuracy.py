#!/usr/bin/env python3
"""
SpectraMambaNet Accuracy Audit — WACV 2027 Zero-Shot Protocol

Finds best/last checkpoint, runs zero-shot evaluation on all 5 test datasets,
compares against SOTA from referenced papers, and saves a JSON audit report.
Optionally queries the local LanceDB paper index for SOTA context excerpts.

Usage:
    !python src/audit_accuracy.py                          # auto-find last ckpt
    !python src/audit_accuracy.py --best                   # auto-find best ckpt
    !python src/audit_accuracy.py --ckpt path/to/f.ckpt   # explicit path
    !python src/audit_accuracy.py --best --tta             # + test-time augmentation
    !python src/audit_accuracy.py --best --query-rag       # + paper excerpts
    !python src/audit_accuracy.py --test-root /path/to/TestDataset

    # No checkpoint / GPU / TestDataset needed — reports all selection rules off the history:
    python src/audit_accuracy.py --history-only --history runs/ours_seed0_history.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# ZERO-SHOT protocol only. Our model trains on Kvasir-SEG + CVC-ClinicDB train images and
# tests on all 5 sets WITHOUT fine-tuning — CVC-300/ColonDB/ETIS are unseen. The target-to-beat
# is the best PraNet-protocol *zero-shot* mDice, sourced from FARF-Net (Li et al., WACV 2026)
# Table 1, which uses this exact protocol and explicitly marks CVC-300/ColonDB/ETIS as unseen
# (repo evidence: papers/Li_FARF-Net_..._WACV_2026_paper.pdf p7).
#
# BAMPolyp's numbers (Kvasir 0.9380, ClinicDB 0.9437, ColonDB 0.9255) are IN-DOMAIN 80/10/10
# per-dataset splits — NOT comparable to zero-shot and DELIBERATELY EXCLUDED as targets.
# Secondary zero-shot references for context (not the ceiling): Polyp-PVT (Kvasir 0.917,
# ClinicDB 0.937, ColonDB 0.808, ETIS 0.787) and PraNet (Kvasir 0.898, ClinicDB 0.899,
# CVC-300 0.871, ColonDB 0.709, ETIS 0.628).
SOTA_TABLE = {
    "CVC-ClinicDB": {
        "mDice": 0.945,
        "source": "FARF-Net (Li et al., WACV 2026), Table 1 — seen split",
    },
    "Kvasir": {
        "mDice": 0.932,
        "source": "FARF-Net (Li et al., WACV 2026), Table 1 — seen split",
    },
    "CVC-300": {
        "mDice": 0.916,
        "source": "FARF-Net (Li et al., WACV 2026), Table 1 — unseen/zero-shot",
    },
    "CVC-ColonDB": {
        "mDice": 0.814,
        "source": "FARF-Net (Li et al., WACV 2026), Table 1 — unseen/zero-shot",
    },
    "ETIS-LaribPolypDB": {
        "mDice": 0.815,
        "source": "FARF-Net (Li et al., WACV 2026), Table 1 — unseen/zero-shot",
    },
}

# Polyp-PVT (Dong et al., CAAI AIR 2023) — the parity bar. Same PVTv2-B2 encoder as ours,
# so any deficit against it is decoder/recipe, not backbone. ColonDB 0.808 / mIoU 0.727 are
# confirmed against the primary source (Polyp-PVT-main/README.md:24); the remaining four are
# the widely-reproduced Table 2 numbers.
#
# Caveat that belongs in the paper, not just here: Polyp-PVT's own Train.py:110-121 scores all
# five benchmark splits every epoch and checkpoints on the best, so its published figures are
# best-epoch-on-test statistics. Our own numbers select on a held-out slice of the 1450 train
# images instead, which is the stricter protocol.
POLYP_PVT_TABLE = {
    "Kvasir":            0.917,
    "CVC-ClinicDB":      0.937,
    "CVC-300":           0.900,
    "CVC-ColonDB":       0.808,
    "ETIS-LaribPolypDB": 0.787,
}

# Phase-1 parity gate: match or beat Polyp-PVT. The two hardest unseen splits carry a small
# reproduction tolerance because they are the ones the acceptance gate is measured on.
#
# IMPORTANT — these are NOT apples-to-apples with POLYP_PVT_TABLE and must not be presented as
# if they were. Polyp-PVT's published numbers come from training on all 1450 images; ours are
# produced with --val-fraction 0.05 held out (1377 images, minus ~1 tail batch/epoch from
# drop_last), because the alternative is selecting checkpoints on the benchmark we report.
# The gate is only meaningful when the acceptance run (--model polyp_pvt) is trained at the
# SAME --val-fraction and --split-seed as the parity run, so both sides lose the same data;
# the comparison is then ours-vs-our-reproduction, with POLYP_PVT_TABLE as published context.
# Never compare a val-held-out run of ours directly against the published full-data number and
# call the difference a result.
TARGET_THRESHOLDS = {
    "Kvasir":            {"mDice": 0.917},
    "CVC-ClinicDB":      {"mDice": 0.937},
    "CVC-300":           {"mDice": 0.900},
    "CVC-ColonDB":       {"mDice": 0.800},
    "ETIS-LaribPolypDB": {"mDice": 0.777},
}


def find_checkpoint(mode: str, explicit_path: str | None) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            sys.exit(f"Checkpoint not found: {explicit_path}")
        return explicit_path

    log_root = Path("lightning_logs")
    if not log_root.exists():
        sys.exit("lightning_logs/ not found. Run training first or pass --ckpt explicitly.")

    ckpt_files = list(log_root.glob("**/checkpoints/*.ckpt"))
    if not ckpt_files:
        sys.exit("No .ckpt files found under lightning_logs/. Run training first.")

    if mode == "last":
        last_ckpts = [p for p in ckpt_files if p.name == "last.ckpt"]
        if last_ckpts:
            return str(sorted(last_ckpts, key=lambda p: p.stat().st_mtime)[-1])
        return str(sorted(ckpt_files, key=lambda p: p.stat().st_mtime)[-1])

    # mode == "best": ModelCheckpoint(monitor="val_dice") writes best-val-{epoch}-{val_dice}.ckpt.
    # Scope the search to the most-recently-modified version directory — if lightning_logs/ is
    # reused across architecture-varying runs without --fresh, an unscoped whole-tree search can
    # silently resolve to a same-shaped checkpoint from an unrelated older experiment.
    version_dirs = [p for p in log_root.glob("version_*") if p.is_dir()]
    scoped_ckpts = []
    if version_dirs:
        latest_version = max(version_dirs, key=lambda p: p.stat().st_mtime)
        scoped_ckpts = list(latest_version.glob("checkpoints/*.ckpt"))
        if scoped_ckpts:
            print(f"Scoping checkpoint search to latest run: {latest_version}")

    search_pool = scoped_ckpts or ckpt_files
    if not scoped_ckpts and version_dirs:
        print("WARNING: no checkpoints found in the latest run directory — falling back to "
              "searching all of lightning_logs/, which may pick up a checkpoint from an "
              "unrelated older run. Pass --ckpt explicitly to avoid ambiguity.")

    best_ckpts = [p for p in search_pool if p.name.startswith("best-val")]
    if best_ckpts:
        # The val_dice is encoded in the filename by ModelCheckpoint; parse it rather than
        # trusting mtime, which orders by write time, not by score.
        def _val_dice(p):
            try:
                return float(p.stem.rsplit("=", 1)[-1])
            except ValueError:
                return -1.0
        return str(max(best_ckpts, key=_val_dice))

    # Legacy runs selected on the test sets and wrote best-zeroshot.ckpt. Those numbers are not
    # comparable to anything published — refuse to resolve to one silently.
    if any(p.name == "best-zeroshot.ckpt" for p in search_pool):
        print("WARNING: only legacy best-zeroshot.ckpt found. That checkpoint was selected by "
              "scoring the five benchmark splits during training, so any number derived from it "
              "is a max-over-checkpoints statistic on its own test set. Pass --ckpt explicitly "
              "if you really want it.")

    print("No best-val checkpoint found — falling back to the most recent snapshot.")
    return str(sorted(search_pool, key=lambda p: p.stat().st_mtime)[-1])


def load_model_and_meta(ckpt_path: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from train import SpectraMambaLightning

    print(f"Loading checkpoint: {ckpt_path}")
    lightning_module = SpectraMambaLightning.load_from_checkpoint(ckpt_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_module = lightning_module.to(device)
    lightning_module.eval()

    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt_data.get("epoch", "?")

    from datetime import datetime as _dt
    ckpt_mtime = _dt.fromtimestamp(os.path.getmtime(ckpt_path)).isoformat(timespec="seconds")

    meta = {"ckpt_path": ckpt_path, "epoch": epoch, "ckpt_mtime": ckpt_mtime}
    return lightning_module, meta


def query_rag_context(dataset_name: str, lancedb_path: str, n: int = 3) -> list[str]:
    try:
        import lancedb
        db = lancedb.connect(lancedb_path)
        table = db.open_table("chunks")
        query = f"{dataset_name} mDice Dice polyp segmentation performance"
        results = table.search(query).limit(n).to_list()
        return [r.get("text", r.get("chunk", "")) for r in results]
    except Exception as e:
        return [f"[RAG unavailable: {e}]"]


def print_audit_table(results: dict, ckpt_meta: dict, use_rag: bool, lancedb_path: str):
    print("\n" + "=" * 78)
    print("  SpectraMambaNet Accuracy Audit — WACV 2027 Zero-Shot Protocol")
    print("=" * 78)
    print(f"  Checkpoint : {ckpt_meta['ckpt_path']}")
    print(f"  Epoch      : {ckpt_meta['epoch']}")
    print(f"  Saved      : {ckpt_meta.get('ckpt_mtime', '?')}")
    print("=" * 78)

    col1, col2 = 22, 7
    hdr = (
        f"{'Dataset':<{col1}} {'mDice':>{col2}} {'mIoU':>{col2}} {'Sm':>{col2}}"
        f" {'wFm':>{col2}} {'MAE':>{col2}} {'Recall':>{col2}} {'HD95':>{col2}}"
        f" {'P-PVT':>{col2}} {'vsPPVT':>{col2+1}} {'SOTA':>{col2}} {'vsSOTA':>{col2+1}} Status"
    )
    print(hdr)
    print("-" * len(hdr))

    summary_pass, summary_fail = [], []

    for ds_name in sorted(results.keys()):
        m = results[ds_name]
        sota_dice = SOTA_TABLE.get(ds_name, {}).get("mDice")
        ppvt_dice = POLYP_PVT_TABLE.get(ds_name)
        target = TARGET_THRESHOLDS.get(ds_name, {}).get("mDice")

        sota_gap = f"{m['mDice'] - sota_dice:+.4f}" if sota_dice is not None else ""
        ppvt_gap = f"{m['mDice'] - ppvt_dice:+.4f}" if ppvt_dice is not None else ""

        status = "  --  "
        if target is not None:
            if m["mDice"] >= target:
                status, _ = "PASS", summary_pass.append(ds_name)
            else:
                status, _ = "FAIL", summary_fail.append(ds_name)

        sota_str = f"{sota_dice:.4f}" if sota_dice else "  --  "
        ppvt_str = f"{ppvt_dice:.4f}" if ppvt_dice else "  --  "
        hd95 = m.get("HD95", float("nan"))
        hd95_str = f"{hd95:.2f}" if hd95 == hd95 else "n/a"

        print(
            f"{ds_name:<{col1}} {m['mDice']:>{col2}.4f} {m['mIoU']:>{col2}.4f}"
            f" {m['Sm']:>{col2}.4f} {m['wFm']:>{col2}.4f} {m['MAE']:>{col2}.4f}"
            f" {m.get('Recall', float('nan')):>{col2}.4f} {hd95_str:>{col2}}"
            f" {ppvt_str:>{col2}} {ppvt_gap:>{col2+1}}"
            f" {sota_str:>{col2}} {sota_gap:>{col2+1}} {status}"
        )

        if use_rag:
            print(f"\n  [RAG context — {ds_name}]")
            for excerpt in query_rag_context(ds_name, lancedb_path):
                snippet = excerpt[:200].replace("\n", " ")
                print(f"    … {snippet}")
            print()

    print("=" * len(hdr))
    if summary_pass:
        print(f"  PASS ({len(summary_pass)}): {', '.join(summary_pass)}")
    if summary_fail:
        print(f"  FAIL ({len(summary_fail)}): {', '.join(summary_fail)}")
    print()


def save_json(results: dict, ckpt_meta: dict, out_path: str):
    audit = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": ckpt_meta,
        "datasets": {},
    }
    for ds_name, m in results.items():
        sota = SOTA_TABLE.get(ds_name, {})
        sota_dice = sota.get("mDice")
        ppvt_dice = POLYP_PVT_TABLE.get(ds_name)
        target_dice = TARGET_THRESHOLDS.get(ds_name, {}).get("mDice")
        audit["datasets"][ds_name] = {
            **m,
            "sota_mDice": sota_dice,
            "sota_source": sota.get("source"),
            "gap_vs_sota": (m["mDice"] - sota_dice) if sota_dice is not None else None,
            "polyp_pvt_mDice": ppvt_dice,
            "gap_vs_polyp_pvt": (m["mDice"] - ppvt_dice) if ppvt_dice is not None else None,
            "target_mDice": target_dice,
            "pass": bool(m["mDice"] >= target_dice) if target_dice is not None else None,
        }

    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Results saved to: {out_path}")


def print_selection_rule_delta(results: dict, history_path: str):
    """Report the same run under both checkpoint-selection rules.

    Our headline number comes from the checkpoint with the best val_dice on a held-out
    slice of the training set — one blind measurement on each benchmark split. Polyp-PVT
    (Train.py:110-121) instead scores all five benchmark splits every epoch and keeps the
    best, so its published numbers are a max-over-epochs on the test sets themselves.
    Comparing ours to theirs directly understates ours by whatever that selection is worth.
    This prints the difference so the comparison can be stated honestly rather than assumed
    away in either direction.
    """
    import json
    try:
        with open(history_path) as f:
            history = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"\n[selection-rule] could not read {history_path}: {e}")
        return
    if not history:
        print(f"\n[selection-rule] {history_path} is empty — was --zeroshot-every set?")
        return

    print("\n" + "=" * 78)
    print("  Same run, both checkpoint-selection rules")
    print(f"  ({len(history)} monitored epochs from {history_path})")
    print("=" * 78)
    print(f"  {'Dataset':<22} {'val-selected':>13} {'best-on-test':>13} {'inflation':>11}  epoch")
    print("-" * 78)

    inflations = []
    for ds in sorted(results):
        ours = results[ds]["mDice"]
        scored = [(h["splits"][ds]["mDice"], h["epoch"]) for h in history if ds in h["splits"]]
        if not scored:
            continue
        peak, peak_epoch = max(scored)
        inflations.append(peak - ours)
        print(f"  {ds:<22} {ours:>13.4f} {peak:>13.4f} {peak - ours:>+11.4f}  {peak_epoch}")

    if inflations:
        print("-" * 78)
        print(f"  Mean inflation from selecting on the test sets: "
              f"{sum(inflations) / len(inflations):+.4f} mDice")
    print("=" * 78)


def _split_mean(entry, names):
    """Mean mDice over `names`, or None if the entry is missing any of them."""
    vals = [entry["splits"][n]["mDice"] for n in names if n in entry["splits"]]
    return sum(vals) / len(vals) if len(vals) == len(names) else None


def _load_history(history_path: str):
    with open(history_path) as f:
        history = json.load(f)
    if not history:
        sys.exit(f"{history_path} is empty — was --zeroshot-every set?")
    return history


IN_DOMAIN_SPLITS = ("Kvasir", "CVC-ClinicDB")


def report_history_only(history_path: str, plateau_n: int = 6, stride: int = 1):
    """Report every checkpoint-selection rule from the training history alone.

    Needs no checkpoint, no GPU and no TestDataset — the five splits were already scored
    at every monitored epoch, so the like-for-like comparison against published numbers is
    a re-read of data we have, not a new measurement.

    The four rules, weakest-selection first:
      last     — the final monitored epoch.
      plateau  — mean over the last `plateau_n` monitored epochs; what we have been quoting.
      val      — the epoch with the highest val_dice on the held-out slice of the 1450
                 TRAINING images. This is what `ModelCheckpoint(monitor="val_dice")` picks
                 and the only rule here that never touches a benchmark split.
      test/mean— the epoch with the highest mean mDice over the five benchmark splits, then
                 read across. One checkpoint, chosen on the test sets: Polyp-PVT's
                 Train.py:110-121 and HGM's MyTrain.py:63-75 both do exactly this, so their
                 published numbers live in this column.
      test/split— per-split maximum over epochs, i.e. a different checkpoint per dataset.
                 An upper bound, not a shippable model. BAMPolyp reports this shape
                 (its Fig. 4 marks "the best epoch for each dataset").
      indomain — the epoch with the highest mean mDice over Kvasir + CVC-ClinicDB only, then
                 read across. This is what `--select-on indomain` actually does at training
                 time, so for any run launched that way it is THE reportable column and
                 test/mean is not. It leaves the three zero-shot splits blind.

    `stride` subsamples to `epoch % stride == 0` before any of that, which answers "what
    would this run have scored on a coarser peek grid" without retraining. Measured on the
    two per-epoch archives, stride 5 costs ~0.004 ZS-3 against stride 1 under the indomain
    rule, so it is a robustness read-out, not an alternative worth running.
    """
    history = _load_history(history_path)
    if stride > 1:
        subsampled = [h for h in history if h["epoch"] % stride == 0]
        if not subsampled:
            sys.exit(f"--history-stride {stride} left no epochs in {history_path}.")
        print(f"\n[stride {stride}] {len(history)} monitored epochs -> {len(subsampled)} candidates.")
        history = subsampled
    splits = sorted({ds for h in history for ds in h["splits"]})
    epochs = [h["epoch"] for h in history]
    tail = history[-plateau_n:]

    indom_scored = [(_split_mean(h, IN_DOMAIN_SPLITS), -h["epoch"]) for h in history]
    indom_scored = [s for s in indom_scored if s[0] is not None]
    indom_row = None
    if indom_scored:
        indom_epoch = -max(indom_scored)[1]
        indom_row = next(h for h in history if h["epoch"] == indom_epoch)

    val_scored = [(h["val_dice"], -h["epoch"]) for h in history if h.get("val_dice") is not None]
    val_epoch = -max(val_scored)[1] if val_scored else None
    val_row = next((h for h in history if h["epoch"] == val_epoch), None)

    mean_scored = [(h.get("mean_mDice") or 0.0, -h["epoch"]) for h in history]
    best_mean_epoch = -max(mean_scored)[1]
    best_mean_row = next(h for h in history if h["epoch"] == best_mean_epoch)

    gaps = sorted(set(range(min(epochs), max(epochs) + 1)) - set(epochs))

    print("\n" + "=" * 92)
    print("  Checkpoint-selection rules - read off the training history, no GPU")
    print(f"  {history_path}: {len(history)} monitored epochs, {min(epochs)}-{max(epochs)}")
    if gaps:
        print(f"  WARNING: {len(gaps)} epochs missing; every max below may be understated.")
    print("=" * 92)
    hdr = (f"  {'Dataset':<20} {'last':>8} {'plateau':>8} {'val':>8} {'indomain':>9} "
           f"{'test/mean':>10} {'test/split':>11}  {'@ep':>4}")
    print(hdr)
    print("-" * 92)

    cols = {k: [] for k in ("last", "plateau", "val", "indomain", "test_mean", "test_split")}
    for ds in splits:
        scored = [(h["splits"][ds]["mDice"], h["epoch"]) for h in history if ds in h["splits"]]
        if not scored:
            continue
        last = scored[-1][0]
        plat = [h["splits"][ds]["mDice"] for h in tail if ds in h["splits"]]
        plateau = sum(plat) / len(plat) if plat else float("nan")
        val = val_row["splits"][ds]["mDice"] if val_row and ds in val_row["splits"] else float("nan")
        indom = (indom_row["splits"][ds]["mDice"]
                 if indom_row and ds in indom_row["splits"] else float("nan"))
        t_mean = (best_mean_row["splits"][ds]["mDice"]
                  if ds in best_mean_row["splits"] else float("nan"))
        t_split, t_split_ep = max(scored)
        for k, v in (("last", last), ("plateau", plateau), ("val", val),
                     ("indomain", indom), ("test_mean", t_mean), ("test_split", t_split)):
            cols[k].append(v)
        print(f"  {ds:<20} {last:>8.4f} {plateau:>8.4f} {val:>8.4f} {indom:>9.4f} "
              f"{t_mean:>10.4f} {t_split:>11.4f}  {t_split_ep:>4}")

    print("-" * 92)
    _COLW = (("last", 8), ("plateau", 8), ("val", 8), ("indomain", 9),
             ("test_mean", 10), ("test_split", 11))
    print(f"  {'MEAN (5 splits)':<20} " + " ".join(
        f"{sum(cols[k]) / len(cols[k]):>{w}.4f}" for k, w in _COLW))

    # The published headline is the zero-shot mean over the three unseen sets only.
    unseen = [d for d in splits if d in ("CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")]
    if len(unseen) == 3:
        idx = [splits.index(d) for d in unseen]
        print(f"  {'ZERO-SHOT (3 unseen)':<20} " + " ".join(
            f"{sum(cols[k][i] for i in idx) / 3:>{w}.4f}" for k, w in _COLW))
    print("=" * 92)

    if indom_row is not None:
        print(f"  indomain gate ({' + '.join(IN_DOMAIN_SPLITS)}) peaked on epoch "
              f"{indom_row['epoch']}. For a run launched with --select-on indomain, the "
              f"ZERO-SHOT figure in that column is the reportable number: the gate never "
              f"sees the three unseen splits, which is the same disclosure class as "
              f"Polyp-PVT's published row (Train.py:116-121).")

    if val_row is not None:
        print(f"  val_dice peaked at {val_row['val_dice']:.4f} on epoch {val_epoch}; "
              f"test/mean peaked on epoch {best_mean_epoch}.")
        if val_epoch != best_mean_epoch:
            delta = (sum(cols["test_mean"]) - sum(cols["val"])) / len(cols["val"])
            print(f"  Selecting on the test sets is worth {delta:+.4f} mDice on the 5-split "
                  f"mean. Any table comparing our 'val' column against a published "
                  f"'test/mean' column is off by roughly that much, in their favour.")
    else:
        print("  No val_dice in this history (--val-fraction 0), so there is no blind "
              "selection rule for this run — report 'last' or 'test/mean', labelled.")

    # PSRC C1/C2: HD95 and Recall are only meaningful at the epoch actually being reported.
    print("-" * 92)
    print(f"  {'Dataset':<20} {'HD95 @last':>11} {'HD95 @t/mean':>13} "
          f"{'Recall @last':>13} {'Recall @t/mean':>15}")
    for ds in splits:
        rows = [h for h in history if ds in h["splits"]]
        if not rows:
            continue
        a, b = rows[-1]["splits"][ds], best_mean_row["splits"].get(ds)
        if b is None:
            continue
        print(f"  {ds:<20} {a['HD95']:>11.1f} {b['HD95']:>13.1f} "
              f"{a['Recall']:>13.4f} {b['Recall']:>15.4f}")
    print("=" * 92)

    _report_ema(history, splits, tail, plateau_n)


def _report_ema(history, splits, tail, plateau_n):
    """The EMA columns, when --ema-decay was on.

    Reported under `last` and `plateau` only, deliberately: the whole point of weight
    averaging here is a number that is NOT a max-over-epochs-on-test statistic, so pairing
    it with a test/mean column would give the labelling constraint straight back.
    """
    ema_rows = [h for h in history if h.get("ema_splits")]
    if not ema_rows:
        return
    if len(ema_rows) < len(history):
        print(f"  NOTE: EMA scored on {len(ema_rows)}/{len(history)} monitored epochs "
              f"(the shadow allocates on the first optimizer step).")

    def col(rows, ds, key):
        vals = [r[key][ds]["mDice"] for r in rows if ds in r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    ema_tail = [h for h in tail if h.get("ema_splits")][-plateau_n:]
    print(f"  {'Dataset':<20} {'raw last':>10} {'EMA last':>10} {'raw plat':>10} "
          f"{'EMA plat':>10} {'d(last)':>9}")
    print("-" * 92)
    acc = {k: [] for k in ("raw_last", "ema_last", "raw_plat", "ema_plat")}
    for ds in splits:
        v = {"raw_last": col(ema_rows[-1:], ds, "splits"),
             "ema_last": col(ema_rows[-1:], ds, "ema_splits"),
             "raw_plat": col(ema_tail, ds, "splits"),
             "ema_plat": col(ema_tail, ds, "ema_splits")}
        for k in acc:
            acc[k].append(v[k])
        print(f"  {ds:<20} {v['raw_last']:>10.4f} {v['ema_last']:>10.4f} "
              f"{v['raw_plat']:>10.4f} {v['ema_plat']:>10.4f} "
              f"{v['ema_last'] - v['raw_last']:>+9.4f}")
    print("-" * 92)
    unseen = [d for d in splits if d in ("CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")]
    if len(unseen) == 3:
        idx = [splits.index(d) for d in unseen]
        m = {k: sum(acc[k][i] for i in idx) / 3 for k in acc}
        print(f"  {'ZERO-SHOT (3 unseen)':<20} {m['raw_last']:>10.4f} {m['ema_last']:>10.4f} "
              f"{m['raw_plat']:>10.4f} {m['ema_plat']:>10.4f} "
              f"{m['ema_last'] - m['raw_last']:>+9.4f}")
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description="SpectraMambaNet accuracy audit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--best", action="store_true", help="Auto-find best-val-*.ckpt (highest val_dice on the held-out train slice)")
    group.add_argument("--last", action="store_true", help="Auto-find last.ckpt (default)")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path")
    parser.add_argument("--tta", action="store_true",
                         help="Also compute a non-headline TTA Dice@0.5 column (headline mDice/mIoU is always TTA-off for SOTA comparability)")
    parser.add_argument("--test-root", type=str, default=None, help="Override TestDataset path")
    parser.add_argument("--query-rag", action="store_true", help="Query LanceDB for paper excerpts")
    parser.add_argument("--lancedb", type=str, default="lancedb", help="LanceDB directory path")
    parser.add_argument("--out", type=str, default=None, help="JSON output path")
    parser.add_argument("--history", type=str, default=None,
                         help="zeroshot_history.json from training. Also reports the "
                              "best-epoch-on-test number, which is the selection rule "
                              "Polyp-PVT's published results use, for like-for-like comparison.")
    parser.add_argument("--history-only", action="store_true",
                         help="Report every checkpoint-selection rule from --history alone. "
                              "No checkpoint, no GPU, no TestDataset needed.")
    parser.add_argument("--plateau-epochs", type=int, default=6,
                         help="How many trailing monitored epochs the 'plateau' column averages.")
    parser.add_argument("--history-stride", type=int, default=1,
                         help="Keep only epochs where epoch %% N == 0 before selecting. Answers "
                              "'what would a coarser peek grid have picked' from a fine history, "
                              "with no retraining. Measured cost of stride 5 under the indomain "
                              "rule: ~0.004 ZS-3.")
    args = parser.parse_args()

    if args.history_only:
        if not args.history:
            sys.exit("--history-only requires --history <zeroshot_history.json>")
        report_history_only(args.history, args.plateau_epochs, args.history_stride)
        return

    mode = "best" if args.best else "last"
    if mode == "best":
        print("\nNOTE: --best selects on val_dice, computed on a held-out 5% slice of the 1450 "
              "TRAINING images — never on the five benchmark splits below. The numbers reported "
              "here are therefore a single blind measurement on those splits, which is stricter "
              "than Polyp-PVT's own published protocol (its Train.py checkpoints on the best "
              "test-set score).\n")
    ckpt_path = find_checkpoint(mode, args.ckpt)

    lightning_module, meta = load_model_and_meta(ckpt_path)

    from test import evaluate_zero_shot

    test_root = args.test_root or os.environ.get("TEST_DATA_ROOT", "/content/TestDataset")

    results = evaluate_zero_shot(model=lightning_module, use_tta=args.tta, test_root=test_root)
    if not results:
        sys.exit("Evaluation returned no results. Check TestDataset path.")

    print_audit_table(results, meta, args.query_rag, args.lancedb)

    if args.history:
        print_selection_rule_delta(results, args.history)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"audit_results_{ts}.json"
    save_json(results, meta, out_path)


if __name__ == "__main__":
    main()
