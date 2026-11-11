#!/usr/bin/env bash
# The ablation arms of the paper's Section 6: train, predict, score and profile each one.
#
# R0-R3, the cumulative table, are verbatim from logs/ablation_R0-R3.txt (the launcher's
# "Resolved training command" lines 208, 532, 895 and 1258). That sweep ran on commit b8ed1e2 of
# the private development repository; the training code it ran is the code in src/, as
# MANIFEST.md section 2 shows. Same environment as fin2: Python 3.12.13, PyTorch 2.11.0+cu128,
# CUDA 12.8, one NVIDIA L4, 6.7 h for all four arms.
#
# M5 (spectral third axis) and M3 (the raster-scan control) ran on commit 5ab1199 in a session
# whose console log was not kept. Their commands are rebuilt from that commit's sweep cell: R3's
# recipe plus the bounded gain, and --third-axis raster for M3. scripts/verify_release.py checks
# that both build at 34,833,672 parameters.
#
# M0 (no block, decoder widened to --fcc-channel 80) was never trained. The paper prints only its
# parameter count, 34,930,646, which verify_release.py also checks.
#
# Every arm: 40 epochs, seed 0, scored from last.ckpt (epoch 39) with no weight averaging. What
# each arm printed, mean Dice:
#
#   arm  Kvasir  ClinicDB  CVC-300  ColonDB   ETIS   five-set mean
#   R0   .9053   .9244     .8926    .8053     .8056  .8666
#   R1   .9150   .9139     .8896    .7974     .7996  .8631
#   R2   .9159   .9325     .8872    .7988     .7896  .8648
#   R3   .9196   .9226     .8909    .8126     .8188  .8729
#   M5   .9281   .9314     .8890    .8195     .7954  .8727
#   M3   .9177   .9223     .8921    .8210     .8170  .8740
#
# Two runs of one configuration landed 0.0040 apart on the five-set mean, so a rerun should be
# read at that resolution, not cell by cell.
#
# Needs CUDA and mamba-ssm. Data roots as in configs/fin2_train_command.sh. --fresh deletes
# lightning_logs/ (src/train.py:1493), so each arm is scored before the next starts; run this in
# a separate clone if lightning_logs/ holds anything you want, such as the fin2 snapshots.
#
#   bash configs/ablation_commands.sh R3       # one arm
#   bash configs/ablation_commands.sh          # R0 R1 R2 R3 M5 M3
set -euo pipefail

TEST="${TEST_DATA_ROOT:-/content/TestDataset}"
CKPT=lightning_logs/version_0/checkpoints/last.ckpt

COMMON=(--fresh --model ours --encoder pvt_v2_b2 --epochs 40 --seed 0 --precision bf16-mixed
        --lr-schedule cosine --warmup-epochs 5 --bce-unweighted --iou-smooth 1.0
        --ms-mask-interp bilinear)
SELECT=(--val-fraction 0 --select-on indomain --select-weights raw --zeroshot-every 20)
BLOCK=(--mamba-stage h4h8 --mamba-scan raster --mamba-bidirectional)
GAIN=(--freq-gain tanh --freq-gain-scale 1.0 --freq-gain-shared)
LOSS_ON=(--boundary-dice-weight 0.3 --contrast-weight 0.1 --fcc-channel 64)
LOSS_OFF=(--boundary-dice-weight 0.0 --contrast-weight 0.0 --fcc-channel 64)

run_arm() {
    local a=$1 hist="runs/abl_$1_seed0.json"
    local -a train profile
    case "$a" in
    R0) train=("${LOSS_OFF[@]}" --edge-stride 8 --aug light --hat-prob 0.0 "${SELECT[@]}"
               --zeroshot-history "$hist" --no-fcc-decoder --no-fcc-region-calib
               --no-mamba-bottleneck)
        profile=(--edge-stride 8 --no-fcc-decoder --no-fcc-region-calib --no-mamba-bottleneck) ;;
    R1) train=("${LOSS_OFF[@]}" --edge-stride 8 --aug light --hat-prob 0.0 "${SELECT[@]}"
               --zeroshot-history "$hist" --no-fcc-region-calib "${BLOCK[@]}")
        profile=(--edge-stride 8 --no-fcc-region-calib "${BLOCK[@]}") ;;
    R2) train=("${LOSS_ON[@]}" --edge-stride 4 --aug light --hat-prob 0.0 "${SELECT[@]}"
               --zeroshot-history "$hist" --fcc-region-proto --ds-mutation "${BLOCK[@]}")
        profile=(--edge-stride 4 --fcc-region-proto "${BLOCK[@]}") ;;
    R3) train=("${LOSS_ON[@]}" --edge-stride 4 --aug endo --hat-prob 0.5 "${SELECT[@]}"
               --zeroshot-history "$hist" --fcc-region-proto --ds-mutation "${BLOCK[@]}")
        profile=(--edge-stride 4 --fcc-region-proto "${BLOCK[@]}") ;;
    M5) train=("${LOSS_ON[@]}" --fcc-region-proto --edge-stride 4 --ds-mutation --aug endo
               --hat-prob 0.5 "${SELECT[@]}" --zeroshot-history "$hist" "${BLOCK[@]}"
               "${GAIN[@]}")
        profile=(--fcc-region-proto --edge-stride 4 "${BLOCK[@]}" "${GAIN[@]}") ;;
    M3) train=("${LOSS_ON[@]}" --fcc-region-proto --edge-stride 4 --ds-mutation --aug endo
               --hat-prob 0.5 "${SELECT[@]}" --zeroshot-history "$hist" "${BLOCK[@]}"
               "${GAIN[@]}" --third-axis raster)
        profile=(--fcc-region-proto --edge-stride 4 "${BLOCK[@]}" "${GAIN[@]}"
                 --third-axis raster) ;;
    *) echo "unknown arm: $a (expected R0 R1 R2 R3 M5 M3)" >&2; return 1 ;;
    esac

    mkdir -p runs preds
    python -u src/train.py "${COMMON[@]}" "${train[@]}"
    python -u src/test.py --checkpoint "$CKPT" --save-preds "preds/$a" --test-root "$TEST"
    python -u src/score_pred_maps.py --pred-root "preds/$a" --per-image --name "$a" \
        --out "runs/abl_$a.json" --test-root "$TEST"
    python -u src/profile_efficiency.py --encoder pvt_v2_b2 --fcc-channel 64 "${profile[@]}" \
        | tee "runs/abl_${a}_cost.txt"
}

arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(R0 R1 R2 R3 M5 M3)
for a in "${arms[@]}"; do run_arm "$a"; done
