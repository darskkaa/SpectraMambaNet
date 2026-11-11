import argparse
import os
import subprocess
import sys
import zipfile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from data.datamodule import find_image_mask_pairs
from train import SpectraMambaLightning
import numpy as np


# PraNet's public TestDataset.zip, the five splits behind every table in the paper.
# It unpacks to TestDataset/<split>/, so a missing test_root is filled only when its
# last path component is TestDataset.
PRANET_TESTSET_ID = "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao"


def ensure_test_dataset(test_root="/content/TestDataset"):
    """Auto-download PraNet's TestDataset if missing."""
    if os.path.exists(test_root):
        return True

    parent = os.path.dirname(os.path.normpath(test_root)) or "."
    drive_id = os.environ.get("TESTSET_DRIVE_ID", PRANET_TESTSET_ID)
    if not drive_id:
        print(f"⚠️  TestDataset not found at {test_root}. Place the standard PraNet")
        print("    TestDataset there (see README, Datasets), or set TESTSET_DRIVE_ID to a mirror.")
        return False
    print(f"⏬ TestDataset not found. Downloading from Google Drive...")
    os.makedirs(parent, exist_ok=True)
    test_zip = os.path.join(parent, "TestDataset.zip")

    try:
        # `-m gdown`: a pip --user install often leaves the gdown script off PATH. Output is
        # not captured, so Drive's "too many users" quota message reaches the user.
        subprocess.run([sys.executable, "-m", "gdown", drive_id, "-O", test_zip], check=True)
        # zipfile rather than `unzip`, which Windows does not ship.
        with zipfile.ZipFile(test_zip) as zf:
            zf.extractall(parent)
        os.remove(test_zip)
    except Exception as e:
        print(f"⚠️  Could not auto-download TestDataset: {e}")
        print(f"    Fetch TestDataset.zip in a browser from https://drive.google.com/file/d/{drive_id}/view")
        print(f"    and unzip it into {parent}.")
        return False
    if not os.path.isdir(test_root):
        print(f"⚠️  The archive unpacked to {os.path.join(parent, 'TestDataset')}, not "
              f"{test_root}. Pass that folder as --test-root.")
        return False
    print(f"✅ TestDataset downloaded and extracted to {test_root}")
    return True


class NativeGTEvalDataset(Dataset):
    """Yields a model-ready input tensor plus the GT mask at its ORIGINAL resolution.

    The canonical PraNet-family benchmark protocol scores predictions against
    native-resolution ground truth, not the training resolution — matching that
    is required for meanDice/meanIoU to be comparable to the SOTA table. Native
    masks vary in size across images, so this must be used with batch_size=1.
    """
    def __init__(self, root_dir, input_transform):
        self.image_paths, self.mask_paths = find_image_mask_pairs(root_dir)
        self.input_transform = input_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask_native = ((mask > 127).astype(np.uint8)) * 255  # native-res uint8 {0,255}

        transformed = self.input_transform(image=image)
        return {'image': transformed['image'], 'mask_native': mask_native,
                'name': os.path.splitext(os.path.basename(self.image_paths[idx]))[0]}


def tta_predict(model, x, out_size):
    """4-view Test-Time Augmentation: original + H-flip + V-flip + HV-flip.
    Averages in probability space to reduce orientation bias from training distribution.
    Returns a probability map resized to out_size (native GT resolution).
    """
    preds = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            xi = x.clone()
            if hflip:
                xi = torch.flip(xi, dims=[-1])
            if vflip:
                xi = torch.flip(xi, dims=[-2])
            p = model(xi)
            if isinstance(p, (tuple, list)):
                p = p[0]
            p = F.interpolate(p, size=out_size, mode='bilinear', align_corners=False)
            p = torch.sigmoid(p)
            if hflip:
                p = torch.flip(p, dims=[-1])
            if vflip:
                p = torch.flip(p, dims=[-2])
            preds.append(p)
    return torch.stack(preds).mean(dim=0)  # [B, 1, H, W] probability map


def _binary_metrics(pred_u8, gt_native):
    """Recall and HD95 at a fixed 0.5 threshold on the min-max-normalized map.

    The headline mDice/mIoU come from a 256-threshold sweep, which has no single
    binary mask to measure a boundary distance against, so these two are reported
    at the conventional midpoint threshold instead. Both are checklist items C1
    and C2 of the Polyp Segmentation Reporting Checklist (Metrics or Mirage?,
    arXiv 2607.08203) — 25 of the 27 papers it audits report neither.

    HD95 is undefined when exactly one of prediction/GT is empty; those images
    return NaN and are counted separately rather than silently substituted with
    the image diagonal (which flatters models that under-segment).
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred = pred_u8 > 127
    gt = gt_native > 127

    tp = np.logical_and(pred, gt).sum()
    n_pos = gt.sum()
    recall = float(tp / n_pos) if n_pos > 0 else float(pred.sum() == 0)

    if not pred.any() and not gt.any():
        return recall, 0.0
    if not pred.any() or not gt.any():
        return recall, float('nan')

    def surface(m):
        return m ^ binary_erosion(m, iterations=1, border_value=0)

    d_gt = distance_transform_edt(~surface(gt))
    d_pred = distance_transform_edt(~surface(pred))
    dists = np.concatenate([d_gt[surface(pred)], d_pred[surface(gt)]])
    return recall, float(np.percentile(dists, 95))


def _minmax_normalize(prob_np):
    """Per-image min-max normalization (MATLAB mapminmax convention used by
    PraNet's MyTest.py before saving predictions as PNGs). Returns uint8 [0,255].
    """
    p_min, p_max = float(prob_np.min()), float(prob_np.max())
    if p_max > p_min:
        prob_norm = (prob_np - p_min) / (p_max - p_min)
    else:
        # Constant map: PraNet's (res-min)/(max-min+1e-8) collapses to all-zeros. Match it
        # exactly rather than returning the raw gray value, so the metric sweep agrees 1:1.
        prob_norm = np.zeros_like(prob_np)
    return (prob_norm * 255).astype(np.uint8)


def evaluate_zero_shot(checkpoint_path=None, model=None, use_tta=False, test_root=None,
                       save_preds=None, img_size=None, splits=None):
    """WACV 2027 zero-shot benchmark — canonical PraNet-family protocol.

    meanDice/meanIoU are computed with a 256-threshold sweep over per-image
    min-max-normalized predictions at NATIVE ground-truth resolution, with no
    TTA — this is the exact estimator behind every number in the SOTA table
    (PraNet, Polyp-PVT, FARF-Net, BAMPolyp), so results are 1:1 comparable.
    TTA, when requested, is reported as a separate non-headline column.
    """
    # 1. Load trained model
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Must provide either checkpoint_path or model")
        # The checkpoint overwrites every encoder weight, so skip timm's ImageNet download.
        model = SpectraMambaLightning.load_from_checkpoint(checkpoint_path, map_location="cpu",
                                                           pretrained_encoder=False)
        if torch.cuda.is_available():
            model.cuda()

    model.eval()

    # Robust device detection (works from callback AND standalone)
    device = next(model.parameters()).device

    # 2. Model input transform (image only — GT is loaded separately at native res)
    # Resolution comes from the checkpoint unless overridden. Evaluating an --img-size 384
    # model at 352 would silently score it out of distribution, and the resulting number
    # would look like a training failure rather than a harness mismatch.
    if img_size is None:
        img_size = int(getattr(getattr(model, "hparams", None), "img_size", 352) or 352)
    input_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    # 3. Define Standard Zero-Shot Datasets
    import py_sod_metrics

    print(f"--- WACV 2027 Zero-Shot Generalization Protocol (canonical PraNet eval, "
          f"input {img_size}x{img_size}) ---")
    if use_tta:
        print("    [TTA column enabled: 4-view flip ensemble, reported separately from headline meanDice/meanIoU]")
    if test_root is None:
        test_root = os.environ.get("TEST_DATA_ROOT", "/content/TestDataset")

    if not ensure_test_dataset(test_root):
        print(f"Warning: test_root {test_root} not found and could not auto-download. Skipping zero-shot eval.")
        return {}

    results = {}
    test_splits = [d for d in os.listdir(test_root)
                   if os.path.isdir(os.path.join(test_root, d)) and d != "__MACOSX"]
    # `splits` restricts the pass so a selection loop (e.g. soup.py's greedy in-domain
    # scoring) can run without ever computing a zero-shot number — blindness by omission,
    # not by promising not to look.
    if splits is not None:
        test_splits = [d for d in test_splits if d in splits]

    for split in sorted(test_splits):
        split_dir = os.path.join(test_root, split)
        dataset = NativeGTEvalDataset(split_dir, input_transform)
        pred_dir = None
        if save_preds:
            pred_dir = os.path.join(save_preds, split)
            os.makedirs(pred_dir, exist_ok=True)
        # batch_size=1: native GT masks vary in size across images.
        # Windows spawns (rather than forks) workers, which re-imports the caller's module;
        # any entry point without an `if __name__ == "__main__"` guard then recurses instead
        # of evaluating. Stay single-process there so the eval path works off-Colab.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=0 if os.name == 'nt' else 2,
                                 pin_memory=torch.cuda.is_available())

        FM = py_sod_metrics.FmeasureV2(
            metric_handlers={
                "dice": py_sod_metrics.DICEHandler(with_dynamic=True, with_adaptive=False),
                "iou": py_sod_metrics.IOUHandler(with_dynamic=True, with_adaptive=False),
                "fm_beta03": py_sod_metrics.FmeasureHandler(with_dynamic=True, with_adaptive=True, beta=0.3),
            }
        )
        WFM = py_sod_metrics.WeightedFmeasure()
        SM = py_sod_metrics.Smeasure(alpha=0.5)
        EM = py_sod_metrics.Emeasure()
        MAE = py_sod_metrics.MAE()

        tta_dice_scores = []
        recalls, hd95s = [], []

        with torch.no_grad():
            for batch in dataloader:
                x = batch['image'].to(device)
                gt_native = batch['mask_native'][0].numpy()  # (H, W) uint8 {0,255}
                native_size = gt_native.shape

                # Architectures define their own test-time head combination in eval mode:
                # Polyp-PVT sums both head logits (Test.py:38, `res + res1`), the FCC decoder
                # returns its final map, and the legacy chain returns `out`. A tuple here
                # means the model did not collapse its heads, so take the primary one.
                logit = model(x)
                if isinstance(logit, (tuple, list)):
                    logit = logit[0]
                logit = F.interpolate(logit, size=native_size, mode='bilinear', align_corners=False)
                prob_np = torch.sigmoid(logit)[0, 0].cpu().numpy()
                pred_u8 = _minmax_normalize(prob_np)

                recall, hd95 = _binary_metrics(pred_u8, gt_native)
                recalls.append(recall)
                hd95s.append(hd95)

                if pred_dir:
                    # The exact array the metrics are computed from, so the released maps
                    # and the reported numbers cannot drift apart. Same form the PraNet
                    # family releases: native-resolution min-max-normalized uint8, not a
                    # thresholded mask.
                    Image.fromarray(pred_u8).save(
                        os.path.join(pred_dir, f"{batch['name'][0]}.png"))

                FM.step(pred=pred_u8, gt=gt_native)
                WFM.step(pred=pred_u8, gt=gt_native)
                SM.step(pred=pred_u8, gt=gt_native)
                EM.step(pred=pred_u8, gt=gt_native)
                MAE.step(pred=pred_u8, gt=gt_native)

                if use_tta:
                    tta_prob = tta_predict(model, x, out_size=native_size)[0, 0].cpu().numpy()
                    tta_u8 = _minmax_normalize(tta_prob)
                    gt_bool = gt_native > 127
                    pred_bool = tta_u8 > 127
                    inter = np.logical_and(pred_bool, gt_bool).sum()
                    denom = pred_bool.sum() + gt_bool.sum()
                    tta_dice_scores.append(1.0 if denom == 0 else (2.0 * inter) / denom)

        fm_res = FM.get_results()
        mean_dice = float(fm_res["dice"]["dynamic"].mean())
        mean_iou = float(fm_res["iou"]["dynamic"].mean())
        max_fm_beta03 = float(fm_res["fm_beta03"]["dynamic"].max())

        sm = float(SM.get_results()['sm'])
        mae = float(MAE.get_results()['mae'])

        # Both ends of the same threshold curve. PraNet-V2's eval.py reports the MEAN, so
        # meanEm is what stays 1:1 comparable to that reference table — but every published
        # comparison table we cite reports E^max_phi, and quoting our mean against their max
        # understates us against models we are not otherwise comparing wrongly.
        em_res = EM.get_results()['em']
        em_curve = em_res.get('dynamic', em_res.get('curve'))
        mean_em = float(em_curve.mean())
        max_em = float(em_curve.max())

        wfm = float(WFM.get_results()['wfm'])

        recall = float(np.mean(recalls))
        hd95_arr = np.array(hd95s, dtype=float)
        n_undefined = int(np.isnan(hd95_arr).sum())
        hd95 = float(np.nanmean(hd95_arr)) if n_undefined < len(hd95_arr) else float('nan')

        line = (
            f"Dataset: {split.upper():15s} | "
            f"meanDice: {mean_dice:.4f} | meanIoU: {mean_iou:.4f} | "
            f"S-m: {sm:.4f} | mean E-m: {mean_em:.4f} | max E-m: {max_em:.4f} | "
            f"max Fm(β²=0.3): {max_fm_beta03:.4f} | wFm: {wfm:.4f} | MAE: {mae:.4f} | "
            f"Recall@0.5: {recall:.4f} | HD95@0.5: {hd95:.2f}px"
        )
        if n_undefined:
            line += f" (HD95 undefined on {n_undefined}/{len(hd95_arr)} imgs, excluded)"
        results[split] = {
            "mDice": mean_dice, "mIoU": mean_iou,
            "Sm": sm, "meanEm": mean_em, "maxEm": max_em,
            "maxFm_b03": max_fm_beta03,
            "wFm": wfm, "MAE": mae,
            "Recall": recall, "HD95": hd95, "HD95_undefined": n_undefined,
            "n_images": len(hd95_arr),
        }
        if use_tta and tta_dice_scores:
            tta_dice = float(np.mean(tta_dice_scores))
            line += f" | [TTA-Dice@0.5 (non-headline): {tta_dice:.4f}]"
            results[split]["ttaDice_b0.5"] = tta_dice

        print(line)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WACV 2027 Zero-Shot Evaluation Protocol")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained .ckpt file in lightning_logs")
    parser.add_argument("--tta", action="store_true",
                        help="Also compute a TTA Dice@0.5 column (reported separately, not the headline SOTA-comparable metric)")
    parser.add_argument("--test-root", type=str, default=None, help="Override TestDataset path")
    parser.add_argument("--save-preds", type=str, default=None,
                        help="Write each prediction to DIR/<split>/<image>.png at native GT "
                             "resolution. These are the maps the reported metrics are computed "
                             "from, and the ones to release alongside the paper.")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override the model input resolution. Defaults to the checkpoint's "
                             "own --img-size (352 for checkpoints predating that flag), which is "
                             "almost always what you want; pass it only to measure a "
                             "train/test resolution mismatch on purpose.")
    args = parser.parse_args()

    evaluate_zero_shot(args.checkpoint, use_tta=args.tta, test_root=args.test_root,
                       save_preds=args.save_preds, img_size=args.img_size)
