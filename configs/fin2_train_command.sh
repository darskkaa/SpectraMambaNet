#!/usr/bin/env bash
# The fin2 launch command, verbatim from logs/run_final_2.txt:437.
#
# Needs a CUDA GPU with mamba-ssm installed. src/train.py refuses to start without the
# real kernel: the pure-PyTorch fallback in src/model/mamba_ref.py is inference-only.
#
# 160 epochs, ~8.4 h on one NVIDIA L4. Resolves to the 51 settings in fin2_run_config.json.
#
# Data roots default to /content/TrainDataset and /content/TestDataset, the paths the run
# used. Anywhere else, export TRAIN_DATA_ROOT and TEST_DATA_ROOT first, or append
# --train-root and --test-root (both added for this release; see MANIFEST.md section 2).
set -euo pipefail

python -u src/train.py \
  --fresh \
  --model ours \
  --encoder pvt_v2_b2 \
  --epochs 160 \
  --seed 0 \
  --precision bf16-mixed \
  --lr-schedule cosine-restarts \
  --restart-cycle-len 20 \
  --warmup-epochs 5 \
  --bce-unweighted \
  --iou-smooth 1.0 \
  --ms-mask-interp bilinear \
  --boundary-dice-weight 0.3 \
  --contrast-weight 0.1 \
  --fcc-channel 64 \
  --fcc-region-proto \
  --edge-stride 4 \
  --mamba-stage h4h8 \
  --mamba-scan raster \
  --mamba-bidirectional \
  --ds-mutation \
  --aug endo \
  --hat-prob 0.5 \
  --val-fraction 0 \
  --select-on indomain \
  --select-weights raw \
  --select-on-test \
  --zeroshot-every 5 \
  --zeroshot-history runs/fin2_seed0.json

# Then build the shipped weights from the eight cycle-end snapshots:
#
#   python src/soup.py "lightning_logs/**/epoch*.ckpt" \
#       --out fin2_soup8.ckpt --report runs/fin2_soup8.json
#
# soup.py takes the test split location from --test-root or TEST_DATA_ROOT, like train.py.
#
# The glob is epoch*.ckpt, not epoch=*.ckpt — under --val-fraction 0 the filename template
# is "epoch{epoch:03d}", so an epoch=* pattern silently matches nothing.
#
# Greedy averaging, ranked on the two in-domain splits only; the unseen centres are scored
# once, after membership is frozen. The published soup took epochs 80 and 120.
# See MANIFEST.md sections 5 and 6.
