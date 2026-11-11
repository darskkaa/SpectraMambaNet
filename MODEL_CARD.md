# Model card: SpectraMambaNet, `fin2_soup8.ckpt`

The sections follow the Hugging Face model card template. Every number below is taken from
`MANIFEST.md` or `results/`, and `scripts/verify_release.py` re-derives it from the files in
this repository.

## Intended use

**Research only.** The weights exist so that others can reproduce the paper's numbers,
compare against them, and build on the method.

**Not for clinical use.** This model is not a medical device. It has not been clinically
validated, tested prospectively or reviewed by any regulator, and it must not be used for
diagnosis, screening, treatment decisions or any other part of patient care.

**Not for commercial use without checking the data terms.** The weights were trained on
Kvasir-SEG, whose terms forbid commercial use without its providers' prior written
permission (`NOTICE`).

## Model details

| | |
|---|---|
| Developed by | the SpectraMambaNet authors (https://github.com/darskkaa/SpectraMambaNet) |
| Task | binary segmentation of colorectal polyps in single RGB colonoscopy frames |
| Architecture | PVTv2-B2 encoder, two tri-axial state-space blocks at the 1/8 and 1/4 scales, an octave-convolution decoder ported from FCC-Net, and a zero-initialized boundary calibration head |
| Initialization | the encoder starts from PVT's ImageNet PVTv2-B2 weights, loaded through `timm` |
| Paper | *Tri-Axial State-Space Polyp Segmentation with Zero-Initialized Boundary Calibration*, WACV 2027, Applications Track |
| Input | one RGB frame, resized to 352×352 by `src/test.py` |
| Output | a per-pixel polyp map at the frame's original resolution, saved by `src/test.py` as an 8-bit image, min-max normalized and not thresholded |
| Parameters | 34,854,150 at inference. The checkpoint holds 35,088,262, because it keeps the 234,112-parameter contrastive head that only training uses |
| Compute | 16.37 GMACs at 352×352 as traced by `thop`, which is a lower bound; 18.77 with an analytic floor for the selective scans `thop` cannot trace. 26.4 FPS at batch 1 on one NVIDIA L4 |
| File | `fin2_soup8.ckpt` on the `fin2-weights` release, 140,669,680 bytes, SHA-256 `598de21496dbc7f7c0619ea0d2401c7dc28b6d5885c638359d695e0c69401e0b` |
| Format | a PyTorch Lightning checkpoint, which is a pickle; read [SECURITY.md](SECURITY.md) before loading it |
| Code licence | MIT (`LICENSE`). Third-party code keeps its own terms (`NOTICE`) |

## Training

**Data.** PraNet's split: 1,450 image-mask pairs, 900 from Kvasir-SEG and 550 from
CVC-ClinicDB. No validation split was held out (`--val-fraction 0`). The archives are
PraNet's public files, linked from the README, and are not redistributed here.

**Procedure.** One run at seed 0 on one NVIDIA L4 under Google Colab: 160 epochs with a
cosine learning rate restarted every 20 epochs after 5 warm-up epochs, `bf16-mixed`
precision, no EMA, and endoscopy augmentation (`--aug endo --hat-prob 0.5`). Besides the
segmentation loss, the objective carries a boundary-band Dice term (weight 0.3), a pixel
contrastive term (weight 0.1) and deep supervision. The command is
`configs/fin2_train_command.sh` and its 51 resolved settings are
`configs/fin2_run_config.json`. Training took 8.42 h, plus 13.2 min of pre-flight checks.

**Shipped weights.** A weight-space average of the epoch-80 and epoch-120 snapshots, chosen
from the eight snapshots taken at the end of each restart cycle by mean Dice on the two
in-domain test sets. `MANIFEST.md` §6 shows that the rule did not read the three unseen
centres.

## Evaluation

Five public test sets under the PraNet protocol. Kvasir and CVC-ClinicDB come from the same
collections as the training data; CVC-300, CVC-ColonDB and ETIS-LaribPolypDB come from
centres that contributed no training data.

| dataset | images | mDice | mIoU | Recall@0.5 | HD95@0.5 |
|---|---|---|---|---|---|
| Kvasir *(in-domain)* | 100 | 0.9274 | 0.8838 | 0.9193 | 36.56 px |
| CVC-ClinicDB *(in-domain)* | 62 | 0.9386 | 0.8935 | 0.9584 | 6.81 px |
| CVC-300 *(unseen)* | 60 | 0.8980 | 0.8326 | 0.9589 | 13.37 px |
| CVC-ColonDB *(unseen)* | 380 | 0.8308 | 0.7531 | 0.8670 | 39.30 px |
| ETIS-LaribPolypDB *(unseen)* | 196 | 0.8219 | 0.7446 | 0.9202 | 102.95 px |

Mean Dice over the five sets: 0.883335.

mDice and mIoU average each image's score over 256 binarization thresholds and then over the
images, as `src/score_pred_maps.py` computes them. Recall@0.5 and HD95@0.5 binarize each map
at 0.5 and average over the images: Recall is pixel recall, and HD95 the 95th-percentile
Hausdorff distance in pixels at the mask's original resolution, defined on all 798 images.
Per-image scores are in `results/fin2_soup8_scored.json`, and `results/significance_fin2.json`
holds paired tests against the released maps of eleven published methods.

## Limitations

- **CVC-300 is the weakest set.** Its 0.8980 mean Dice is 0.010 below the best published
  encoder-scale value (CTNet, 0.908). On released maps, CTNet and Polyp-PVT both score higher
  there, by 0.0102 and 0.0022, and neither difference is significant.
- **Some frames fail outright.** Per-image Dice is below 0.1 on 1 of 100 Kvasir frames, 8 of
  380 CVC-ColonDB frames and 10 of 196 ETIS-Larib frames, and on none of CVC-ClinicDB or
  CVC-300. The README shows the worst frames above that line.
- **One seed.** Two runs of one configuration landed 0.0040 apart on the five-set mean, so
  differences smaller than that are not resolved.
- **Selection used test data.** With no validation split, snapshots were ranked on the Kvasir
  and CVC-ClinicDB test sets, the convention behind the published numbers this model is
  compared with. The in-domain scores are therefore optimistic. The unseen centres took no
  part in selection.
- **Cost is a lower bound.** 16.37 GMACs omits the selective scans, and 18.77 adds only an
  analytic floor for them. 26.4 FPS was measured on a datacentre GPU at batch 1, not on
  endoscopy-suite hardware.
- **No external or prospective test.** All five test sets are public benchmark collections.
  Their archives hold images and masks only, with no patient, device or imaging-mode
  metadata, so performance across patient groups, endoscopes and imaging modes is unknown.

## Getting started

```bash
curl -LO https://github.com/darskkaa/SpectraMambaNet/releases/download/fin2-weights/fin2_soup8.ckpt
echo "598de21496dbc7f7c0619ea0d2401c7dc28b6d5885c638359d695e0c69401e0b  fin2_soup8.ckpt" | sha256sum -c -
python src/test.py --checkpoint fin2_soup8.ckpt --test-root data/TestDataset --save-preds preds
```

The README covers installation, the datasets, the CPU fallback scan for machines without
`mamba-ssm`, and how to score saved maps.

## Environmental impact

8.42 h of training and 13.2 min of pre-flight on one NVIDIA L4, under Google Colab.
Emissions were not measured.

## Citation and contact

Cite the paper as the README and `CITATION.cff` give it. Questions and problems go to this
repository's issue tracker, and security reports go through [SECURITY.md](SECURITY.md).
