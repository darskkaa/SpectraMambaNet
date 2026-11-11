"""Rebuild a zero-shot history JSON from a saved training console log.

The launcher's atexit backup copies `lightning_logs` only, so a run whose runtime is
recycled loses `runs/*_history.json` — the file `audit_accuracy.py --history` needs to
report the best-epoch-on-test selection rule. Every number in it is also printed to
stdout, so the log is a lossless substitute.

    python src/recover_history.py colab_log.txt runs/ppvt_seed0_history.json
"""
import json
import re
import sys

# test.py prints `split.upper()`; these are the on-disk TestDataset directory names.
CANONICAL = {
    "CVC-300": "CVC-300",
    "CVC-CLINICDB": "CVC-ClinicDB",
    "CVC-COLONDB": "CVC-ColonDB",
    "ETIS-LARIBPOLYPDB": "ETIS-LaribPolypDB",
    "KVASIR": "Kvasir",
}

# `--select-on-test` relabels the block header SELECTING; the numbers underneath are identical.
MONITOR_RE = re.compile(r"Zero-shot (?:MONITORING|SELECTING) at epoch (\d+)")
SPLIT_RE = re.compile(
    r"Dataset:\s+(\S+)\s+\|\s+meanDice:\s+([\d.]+)\s+\|\s+meanIoU:\s+([\d.]+).*?"
    r"Recall@0\.5:\s+([\d.]+)\s+\|\s+HD95@0\.5:\s+([\d.]+|nan)px"
)
# Closes the raw block. Every monitored epoch prints the five splits twice — raw weights
# first, then the EMA shadow under `--- EMA: ZS-3 ...` — and both match SPLIT_RE, so without
# a terminator the EMA block silently overwrites the raw one and the whole recovered history
# is EMA. The headline metric is raw (train.py's --select-weights raw), so raw is what a
# recovered history has to contain.
RAW_END_RE = re.compile(r"---\s*ZS-3 mDice")
# Val Dice is `N/A` whenever --val-fraction 0; the epoch is still recoverable without it.
EPOCH_RE = re.compile(
    r"Epoch (\d+)/\d+\s+\|\s+Train Loss:\s+[\d.]+\s+\|\s+Val Dice:\s+([\d.]+|N/A)"
)


def parse(lines):
    # The `Epoch NNN` summary is printed *before* that epoch's zero-shot block, and NNN is
    # 0-indexed while the block header is 1-indexed. Stash by 1-indexed epoch and join on
    # flush; closing a block on the next Epoch line instead attributes epoch N+1's val dice
    # to epoch N.
    history, cur, val_by_epoch = [], None, {}

    def flush():
        nonlocal cur
        if cur and cur["splits"]:
            cur["val_dice"] = val_by_epoch.get(cur["epoch"])
            cur["mean_mDice"] = sum(s["mDice"] for s in cur["splits"].values()) / len(cur["splits"])
            history.append(cur)
        cur = None

    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            val_by_epoch[int(m.group(1)) + 1] = None if m.group(2) == "N/A" else float(m.group(2))
            continue

        m = MONITOR_RE.search(line)
        if m:
            flush()
            cur = {"epoch": int(m.group(1)), "mean_mDice": None,
                   "val_dice": None, "splits": {}}
            continue
        if cur is None:
            continue
        if RAW_END_RE.search(line):
            flush()
            continue

        m = SPLIT_RE.search(line)
        if m:
            name = CANONICAL.get(m.group(1).rstrip("|").strip())
            if name:
                cur["splits"][name] = {
                    "mDice": float(m.group(2)),
                    "mIoU": float(m.group(3)),
                    "Recall": float(m.group(4)),
                    "HD95": float("nan") if m.group(5) == "nan" else float(m.group(5)),
                }

    flush()
    return history


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    log_path, out_path = sys.argv[1], sys.argv[2]
    with open(log_path, encoding="utf-8", errors="replace") as f:
        history = parse(f)

    if not history:
        sys.exit(f"No monitoring blocks parsed from {log_path}.")

    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)

    epochs = [h["epoch"] for h in history]
    missing = sorted(set(range(min(epochs), max(epochs) + 1)) - set(epochs))
    print(f"Recovered {len(history)} monitored epochs ({min(epochs)}-{max(epochs)}) -> {out_path}")
    if missing:
        print(f"  WARNING: {len(missing)} epochs absent from the log "
              f"(truncated console scrollback): {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")
        print("  The best-on-test peak may therefore be understated.")

    for ds in sorted(history[0]["splits"]):
        scored = [(h["splits"][ds]["mDice"], h["epoch"]) for h in history if ds in h["splits"]]
        peak, epoch = max(scored)
        print(f"  {ds:<20} peak {peak:.4f} @ epoch {epoch:<4} final {scored[-1][0]:.4f}")


if __name__ == "__main__":
    main()
