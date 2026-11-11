import contextlib
import math
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from data.datamodule import PolypDataModule
from model.contrast import PixelContrastLoss
from model.net import SpectraMambaNet
from model.polyp_pvt import PolypPVT

torch.set_float32_matmul_precision('medium')  # Speed up Ampere+ GPUs


def _nonempty_subsets(n):
    """All 2^n - 1 non-empty index subsets, as tuples. EMCAD's `powerset` (utils/utils.py:20)
    with the empty set dropped the way its trainer drops it (`if s == []: continue`)."""
    return tuple(tuple(i for i in range(n) if m >> i & 1) for m in range(1, 1 << n))


_N_SUBSETS = {n: (1 << n) - 1 for n in range(1, 9)}

MODELS = ('ours', 'polyp_pvt')

# FCC-Net's 5x5 edge-GT dilation runs pre-resize at each image's native resolution
# (their data_val.py:152-153), so its realized edge thickness varies per image; we fix the
# radius at 2 base-resolution pixels instead. Multi-scale training resizes the GT to
# 256/352/448 before _fcc_loss runs,
# so a hardcoded 5x5 would realize 2.75px at 256 and 1.57px at 448 — the edge target's
# thickness, and its positive-pixel fraction, would depend on which scale a step landed on.
# Same resolution-blind-kernel bug BoundaryRegionCalib.band_kernel was fixed for.
# `ref_size` is the *base* training resolution, not a fixed 352: --img-size 384 moves the
# multi-scale sizes to 288/384/480, and pinning the reference at 352 would then define the
# band as 2*352/384 = 1.83 base-resolution px instead of 2. It is the same quantity
# BoundaryRegionCalib.input_size holds, and both track --img-size for the same reason.
EDGE_DILATE_PX = 2
EDGE_DILATE_REF_SIZE = 352


def edge_dilate_kernel(gt_h, ref_size=EDGE_DILATE_REF_SIZE):
    r = max(1, int(round(EDGE_DILATE_PX * gt_h / ref_size)))
    return 2 * r + 1


class ModelEma:
    """Exponential moving average of a model's weights, held outside the nn.Module tree.

    Deliberately not an nn.Module: an attribute holding a second copy of the network would
    show up in `self.parameters()`, so `configure_optimizers` would hand the shadow weights
    to AdamW and train them.

    Buffers (BatchNorm `running_mean`/`running_var`/`num_batches_tracked`) are **copied,
    not blended**. They are already running averages with their own momentum; blending
    them averages an average, and `num_batches_tracked` is an integer counter for which a
    convex combination is meaningless.

    The shadow is allocated lazily on the first update so it lands on whatever device
    Lightning has already moved the model to.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.param_keys = {k for k, _ in model.named_parameters()}
        self.state = None

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        if self.state is None:
            self.state = {k: v.detach().clone().float() if k in self.param_keys
                          else v.detach().clone()
                          for k, v in sd.items()}
            return
        # A resumed run reloads the shadow from the checkpoint on CPU while the model is
        # already on the GPU; mul_/add_ across devices would raise on the first step.
        for k, v in sd.items():
            if self.state[k].device != v.device:
                self.state = {kk: vv.to(v.device) for kk, vv in self.state.items()}
            break
        d = self.decay
        for k, v in sd.items():
            if k in self.param_keys:
                self.state[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.state[k].copy_(v.detach())


class EpochLoggingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        def fmt(key):
            v = trainer.callback_metrics.get(key)
            return f"{v.item():.4f}" if v is not None else "N/A"
        print(f"Epoch {trainer.current_epoch:03d}/{trainer.max_epochs} | "
              f"Train Loss: {fmt('train_loss')} | Val Dice: {fmt('val_dice')}")
        # self.log() only reaches the Lightning logger's own files, and every archived run
        # log in this repo is a captured stdout stream — which is why not one of them carries
        # a tel/ value, and why the question "did the Mamba block ever engage?" has never had
        # an answer. Print them next to the loss so the archive keeps them.
        tel = {k: v for k, v in trainer.callback_metrics.items() if k.startswith("tel/")}
        if tel:
            print("  [tel] " + "  ".join(f"{k[4:]}={v.item():.3e}" for k, v in sorted(tel.items())))


class AMPScalerTelemetryCallback(Callback):
    """Logs the AMP GradScaler's scale each epoch. A drop means the scaler backed off
    (hit inf/nan) at least once since the last epoch — otherwise a silent skipped
    optimizer step is indistinguishable from "gradient signal is genuinely small",
    which matters here since the spectral/Mamba paths are known to be NaN-prone."""

    def __init__(self):
        self._prev_scale = None

    def on_train_epoch_end(self, trainer, pl_module):
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is None:
            return  # no scaler on CPU/bf16 runs — nothing to report
        scale = scaler.get_scale()
        if self._prev_scale is not None and scale < self._prev_scale:
            approx_skips = round(math.log2(self._prev_scale / scale))
            print(f"[amp] scaler_scale={scale:.1f} (prev={self._prev_scale:.1f}) "
                  f"-- BACKED OFF, ~{approx_skips} skipped step(s) this epoch")
        else:
            print(f"[amp] scaler_scale={scale:.1f} (prev={self._prev_scale})")
        # A scale that only ever decays means the *unscaled* gradient is itself above fp16
        # max, which backing off cannot fix — the scaler just walks to zero and the run NaNs.
        # W1 died this way: 16384 -> 64 over 38 epochs, then underflow at epoch 39.
        if scale < 128:
            print(f"[amp] WARNING: scale {scale:.3g} < 128 and only ever decaying. This is "
                  f"the W1 failure signature; expect NaN within ~5 epochs. Find the "
                  f"overflowing gradient rather than waiting it out.")
        self._prev_scale = scale


class ZeroShotMonitorCallback(Callback):
    """Periodically prints the full 5-split zero-shot score. MONITORING ONLY.

    This callback deliberately does NOT write a checkpoint. It used to select
    `best-zeroshot.ckpt` by mean mDice over the same five splits that get
    reported, which makes every published number a max-over-checkpoints
    statistic on its own test set. Checkpoint selection now comes from
    `val_dice` on a held-out slice of the 1450 training images (see
    `PolypDataModule.make_val_split`), and this output is diagnostic context
    only — never a selection signal.
    """

    def __init__(self, every_n_epochs=20, history_path=None, select_on_test=False,
                 select_on="5split", select_weights="auto", test_root=None):
        self.every_n_epochs = every_n_epochs
        self.test_root = test_root
        # Opt-in ONLY (--select-on-test). With --val-fraction 0 there is no blind selection
        # signal at all, and ModelCheckpoint(every_n_epochs=20) will not have saved a file at
        # whichever epoch the run actually peaked on. Every paper in our comparison table
        # (Polyp-PVT Train.py:110-121, HGM MyTrain.py:63-75) selects exactly this way, so
        # reproducing their protocol means writing this checkpoint — but a number taken from
        # it is a max-over-epochs-on-test statistic and must be labelled as such in any table.
        self.select_on_test = select_on_test
        # WHICH score selects. Polyp-PVT does NOT select on the five benchmark splits: it
        # scores them for logging (Train.py:111-115) and then selects on a *separate* call,
        # `test(model, test_path, 'test')` (Train.py:116-121), against a `test/` folder whose
        # own argparse help calls it "the testing Kvasir dataset" — i.e. the in-domain
        # held-out set, not CVC-300/ColonDB/ETIS. So the published protocol never peeks at
        # the zero-shot splits, and `5split` (our historical rule) is strictly *more*
        # test-peeked than the papers we compare to. `indomain` is the matching rule and
        # leaves ZS-3 blind. `5split` is kept only so archived runs stay comparable.
        self.select_on = select_on
        # WHICH WEIGHTS the score comes from. 'auto' selects on the EMA score whenever an EMA
        # exists, which is right only if EMA is what the paper reports. Decision 2026-08-08
        # made the headline **raw** (every published table we compare to is raw weights;
        # sota_positioning.md §0 axis 3), so a run reporting raw must select on raw — otherwise
        # best-on-test.ckpt is the peak of a curve nobody reads and its raw weights are off-peak.
        self.select_weights = select_weights
        self.best_mean = -1.0
        # Appending the per-epoch scores lets the best-epoch-on-test number be recovered
        # after the fact. Polyp-PVT's Train.py:110-121 scores all five benchmark splits
        # every epoch and checkpoints on the best, so its published 0.808 is a
        # max-over-epochs-on-test statistic; reproducing it requires the same history.
        # Recording it is not the same as selecting on it — no checkpoint is written here.
        self.history_path = history_path
        # Neither field survives a crash on its own: this callback is reconstructed by
        # __init__ on resume, and Lightning restores no state into it. That silently cost two
        # things at once — the first monitored epoch reopened history_path in "w" mode and
        # truncated every earlier epoch, and best_mean back at -1.0 let that same epoch
        # overwrite best-on-test.ckpt with weights worse than the pre-crash peak. Replaying
        # the JSON restores both, so a resumed run and an uninterrupted one agree.
        self.history = self._load_history(history_path)
        if self.history:
            self.best_mean = max((s for s in map(self._select_score, self.history)
                                  if s is not None), default=-1.0)
            print(f"--- Zero-shot monitor resumed: {len(self.history)} epochs replayed from "
                  f"{history_path}, best {select_on} so far {self.best_mean:.4f} ---")

    @staticmethod
    def _load_history(path):
        if not path or not os.path.exists(path):
            return []
        import json
        with open(path) as f:
            return json.load(f)

    def _select_score(self, entry):
        key = "mean_mDice" if self.select_on == "5split" else "indomain_mDice"
        # Mirrors the live precedence below: 'raw' never looks at the shadow, 'auto' prefers
        # it and falls back. Replaying under a different rule would pick a different peak.
        if self.select_weights != "raw" and entry.get("ema_" + key) is not None:
            return entry["ema_" + key]
        return entry.get(key)

    def _evaluate(self, pl_module):
        from test import evaluate_zero_shot
        was_training = pl_module.training
        try:
            return evaluate_zero_shot(model=pl_module, test_root=self.test_root)
        finally:
            if was_training:
                pl_module.train()

    # The three splits the PraNet protocol never trains on. Kvasir and CVC-ClinicDB are
    # in-domain held-out, so `mean_mDice` over all five is not a zero-shot number however it
    # is labelled — it is ~0.02 higher, because the two easy in-domain splits are in it.
    ZS3 = ("CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")

    # The two splits drawn from the same centres as the 1450 training images. These are the
    # only ones the published selection rule is allowed to see.
    IN_DOMAIN = ("Kvasir", "CVC-ClinicDB")

    @classmethod
    def _indomain(cls, results):
        vals = [results[s]["mDice"] for s in cls.IN_DOMAIN if s in results]
        return sum(vals) / len(vals) if len(vals) == len(cls.IN_DOMAIN) else None

    @staticmethod
    def _splits(results):
        return {k: {m: v[m] for m in ("mDice", "mIoU", "Recall", "HD95")}
                for k, v in results.items()}

    @classmethod
    def _zs3(cls, results):
        zs = [results[s]["mDice"] for s in cls.ZS3 if s in results]
        return sum(zs) / len(zs) if len(zs) == len(cls.ZS3) else None

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs <= 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        tag = "SELECTING" if self.select_on_test else "MONITORING"
        if not self.select_on_test:
            note = "not used for checkpoint selection"
        elif self.select_on == "indomain":
            # Matches the published convention (Polyp-PVT Train.py:116-121 selects on a
            # separate in-domain folder), so ZS-3 never enters selection and needs no label.
            note = "SELECTION ON IN-DOMAIN HELD-OUT - ZS-3 stays blind"
        else:
            note = ("SELECTION ON TEST - label any number from this checkpoint as "
                    "max-over-epochs-on-test")
        print(f"\n--- Zero-shot {tag} at epoch {trainer.current_epoch + 1} ({note}) ---")
        results = self._evaluate(pl_module)
        if not results:
            return
        mean_dice = sum(r["mDice"] for r in results.values()) / len(results)
        zs3 = self._zs3(results)
        indom = self._indomain(results)
        # Both, always, and never the 5-split one alone: the reported headline is ZS-3, and
        # every past run's "mean zero-shot mDice" line had to be re-derived by hand from the
        # per-split rows because the printed number silently included Kvasir and ClinicDB.
        print(f"--- ZS-3 mDice (CVC-300/ColonDB/ETIS): "
              f"{'n/a' if zs3 is None else f'{zs3:.4f}'}  |  mean over all "
              f"{len(results)} splits (incl. in-domain Kvasir+ClinicDB): {mean_dice:.4f} ---")
        entry = {
            "epoch": trainer.current_epoch + 1,
            "zs3_mDice": zs3,
            "mean_mDice": mean_dice,
            "indomain_mDice": indom,
            "val_dice": float(trainer.callback_metrics["val_dice"])
                        if "val_dice" in trainer.callback_metrics else None,
            "splits": self._splits(results),
        }
        # The EMA curve is the point of --ema-decay: its `last` epoch is a legitimate
        # non-test-peeked number, unlike anything read off best-on-test.ckpt. Scored here
        # so the two trajectories are directly comparable epoch by epoch.
        ema_sel = {}
        if getattr(pl_module, "ema", None) is not None and pl_module.ema.state is not None:
            with pl_module.ema_weights():
                ema_results = self._evaluate(pl_module)
            if ema_results:
                ema_mean = sum(r["mDice"] for r in ema_results.values()) / len(ema_results)
                ema_zs3 = self._zs3(ema_results)
                ema_indom = self._indomain(ema_results)
                entry["ema_zs3_mDice"] = ema_zs3
                entry["ema_mean_mDice"] = ema_mean
                entry["ema_indomain_mDice"] = ema_indom
                entry["ema_splits"] = self._splits(ema_results)
                ema_sel = {"5split": ema_mean, "indomain": ema_indom}
                print(f"--- EMA: ZS-3 {'n/a' if ema_zs3 is None else f'{ema_zs3:.4f}'} "
                      f"(delta vs raw {'n/a' if None in (ema_zs3, zs3) else f'{ema_zs3 - zs3:+.4f}'})"
                      f"  |  5-split {ema_mean:.4f} (delta vs raw {ema_mean - mean_dice:+.4f}) ---")
        # A resume rewinds to the checkpoint's epoch, which can be earlier than the last epoch
        # in the replayed JSON. Dropping the overshoot keeps the file strictly increasing in
        # epoch instead of carrying two entries for the re-run epochs.
        self.history = [e for e in self.history if e.get("epoch", 0) < entry["epoch"]]
        self.history.append(entry)
        if self.history_path:
            import json
            # `--zeroshot-history runs/*.json` names a directory that a fresh clone does not
            # have. Without this the first monitored epoch dies on FileNotFoundError after the
            # full 5-split eval has already been paid for.
            parent = os.path.dirname(self.history_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=2)

        if not self.select_on_test:
            return
        # Select on the weights we will actually report — picking the peak of a curve nobody
        # reads leaves the reported weights off-peak. That cuts both ways: it is the bug that
        # made A1's best-on-test.ckpt carry a non-peak EMA state, and under --select-weights
        # raw it is why an EMA run must NOT select on its shadow.
        raw_sel = {"5split": mean_dice, "indomain": indom}
        if self.select_weights == "raw":
            score, which = raw_sel.get(self.select_on), "raw"
        else:
            score = ema_sel.get(self.select_on) if ema_sel else None
            which = "EMA"
            if score is None:
                score, which = raw_sel.get(self.select_on), "raw"
        if score is None:
            print(f"--- selection skipped: '{self.select_on}' score unavailable "
                  f"(splits present: {sorted(results)}) ---")
            return
        if score <= self.best_mean:
            return
        self.best_mean = score
        # Written next to the ModelCheckpoint files so _save_checkpoints picks it up.
        ckpt_dir = getattr(trainer.checkpoint_callback, "dirpath", None) or "lightning_logs"
        os.makedirs(ckpt_dir, exist_ok=True)
        dest = os.path.join(ckpt_dir, "best-on-test.ckpt")
        trainer.save_checkpoint(dest)
        blind = " (ZS-3 stays blind)" if self.select_on == "indomain" else ""
        print(f"--- Saved {dest} (selected on {which} {self.select_on} {score:.4f}{blind}; "
              f"ZS-3 {'n/a' if zs3 is None else f'{zs3:.4f}'}, 5-split {mean_dice:.4f}, "
              f"epoch {trainer.current_epoch + 1}) ---")


class SpectraMambaLightning(pl.LightningModule):
    """Training wrapper for both the proposed model and the Polyp-PVT baseline.

    Uses manual optimization so the optimizer-step semantics can be matched to a
    reference recipe exactly. Polyp-PVT takes one optimizer step *per multi-scale
    rate* (Train.py:70-90 puts zero_grad/step inside the size_rates loop), which
    is 3x the steps per epoch that a sum-then-backward implementation performs.
    That difference is large enough to matter for reproducing its published
    numbers, so it is configurable via `step_per_scale` rather than assumed.
    """

    def __init__(self, model_name='ours', learning_rate=1e-4, weight_decay=1e-4,
                 max_epochs=100, warmup_epochs=0, lr_schedule='const', restart_cycle_len=20,
                 encoder_lr_scale=1.0,
                 grad_clip_val=0.5, grad_clip_algorithm='value',
                 multi_scale=True, step_per_scale=True, accumulate_grad_batches=1,
                 use_boundary_attn=False, use_wavelet_decoder=False, use_dsra=False,
                 use_fcc_decoder=True, fcc_edge_weight=3.0, fcc_region_calib=True,
                 use_rfp=False, pretrained_encoder=True, fcc_share_bn=False,
                 ds_mutation=False, mamba_bidirectional=False,
                 tversky_weight=0.0, tversky_alpha=0.7, tversky_beta=0.3,
                 boundary_dice_weight=0.0, edge_stride=8, mamba_stage='bottleneck',
                 use_mamba_bottleneck=True, encoder_name='pvt_v2_b2',
                 mamba_scan='axial', fcc_region_proto=False, ema_decay=0.0,
                 fcc_channel=64, fcc_reduce_octave=False, bce_unweighted=False,
                 iou_smooth=1e-5, ms_mask_interp='nearest', img_size=352,
                 fullres_head=False, contrast_weight=0.0, mamba_freq_axis=True,
                 mamba_ls_init=1e-4, freq_gain='clamp', freq_gain_scale=3.0,
                 freq_gain_shared=False, third_axis='freq'):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if model_name not in MODELS:
            raise ValueError(f"--model must be one of {MODELS}, got {model_name!r}")
        if step_per_scale and accumulate_grad_batches != 1:
            raise ValueError("step_per_scale=True steps once per scale, which is incompatible "
                             "with gradient accumulation across batches; set accumulate_grad_batches=1")

        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")

        if ms_mask_interp not in ('nearest', 'bilinear'):
            raise ValueError(f"ms_mask_interp must be 'nearest' or 'bilinear', got {ms_mask_interp!r}")

        # 32 divides the base size AND each multi-scale rate: 0.75*x and 1.25*x are snapped
        # by _scales, but only a base that is itself a multiple of 128 makes all three land
        # exactly on the requested ratio. Requiring 32 is the encoder's hard constraint; the
        # snapping absorbs the rest.
        if img_size % 32:
            raise ValueError(f"img_size must be divisible by 32 (stride-32 encoder + FCC "
                             f"decoder cascade), got {img_size}")

        if model_name == 'polyp_pvt':
            if mamba_scan != 'axial' or fcc_region_proto or not mamba_freq_axis \
                    or mamba_ls_init != 1e-4 or freq_gain != 'clamp' \
                    or freq_gain_scale != 3.0 or freq_gain_shared or third_axis != 'freq':
                raise ValueError("--mamba-scan/--fcc-region-proto/--no-freq-axis/"
                                 "--mamba-ls-init/--freq-gain*/--third-axis reach "
                                 "SpectraMambaNet only; "
                                 "the PolypPVT baseline has neither a scan nor a region calib")
            if fcc_channel != 64 or fcc_reduce_octave or fullres_head:
                raise ValueError("--fcc-channel/--fcc-reduce-octave/--fullres-head size or "
                                 "extend the FCC decoder; the PolypPVT baseline does not "
                                 "have one")
            self.model = PolypPVT(pretrained_encoder=pretrained_encoder,
                                  encoder_name=encoder_name)
        else:
            self.model = SpectraMambaNet(
                encoder_name=encoder_name,
                use_boundary_attn=use_boundary_attn, use_wavelet_decoder=use_wavelet_decoder,
                use_dsra=use_dsra, use_fcc_decoder=use_fcc_decoder,
                fcc_channel=fcc_channel, fcc_reduce_octave=fcc_reduce_octave,
                fcc_region_calib=fcc_region_calib, use_rfp=use_rfp,
                pretrained_encoder=pretrained_encoder, fcc_share_bn=fcc_share_bn,
                ds_mutation=ds_mutation, mamba_bidirectional=mamba_bidirectional,
                edge_stride=edge_stride, mamba_stage=mamba_stage,
                use_mamba_bottleneck=use_mamba_bottleneck,
                mamba_scan=mamba_scan, fcc_region_proto=fcc_region_proto,
                img_size=img_size, fullres_head=fullres_head,
                use_contrast_head=contrast_weight > 0,
                mamba_freq_axis=mamba_freq_axis,
                mamba_ls_init=mamba_ls_init,
                freq_gain=freq_gain,
                freq_gain_scale=freq_gain_scale,
                freq_gain_shared=freq_gain_shared,
                third_axis=third_axis,
            )

        # Built only when the weight is on, so `preds[-1]` is an embedding exactly when this
        # is not None — the tuple is peeled by this flag, never by length (a 6-tuple is
        # ambiguous between --fullres-head and --contrast-weight).
        self.contrast_loss = PixelContrastLoss() if contrast_weight > 0 else None

        self.ema = ModelEma(self.model, ema_decay) if ema_decay > 0 else None

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------ EMA

    def _ema_update(self):
        if self.ema is not None:
            self.ema.update(self.model)

    @contextlib.contextmanager
    def ema_weights(self):
        """Temporarily swap the live weights for the EMA shadow (eval only)."""
        if self.ema is None or self.ema.state is None:
            yield False
            return
        backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema.state)
        try:
            yield True
        finally:
            self.model.load_state_dict(backup)

    def on_save_checkpoint(self, checkpoint):
        # Stored alongside, never in place of, the raw weights: --ema-decay has to stay an
        # attributable ablation, so a checkpoint must carry both trajectories.
        if self.ema is not None and self.ema.state is not None:
            checkpoint['ema_state'] = self.ema.state

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and 'ema_state' in checkpoint:
            self.ema.state = {k: v.clone() for k, v in checkpoint['ema_state'].items()}

    # ---------------------------------------------------------------- losses

    def _seg_loss_pranet(self, pred, mask):
        """wBCE + wIoU — the structure loss used verbatim by PraNet, Polyp-PVT and
        FCC-Net, and the only segmentation loss in this codebase.

        It previously also carried soft-Dice and Tversky(0.3/0.7) terms. All four
        are overlap losses on the same prediction, so they are strongly
        correlated rather than complementary, and the recall-biased Tversky term
        pushes toward over-segmentation, which the min-max-normalized threshold
        sweep in the eval protocol penalises.

        Two deviations from the reference are switchable, because closing them is the
        cheapest hypothesis for our 0.021 Polyp-PVT reproduction gap:

        `bce_unweighted` reproduces the reference *behaviour*, not its source. PraNet,
        Polyp-PVT, CTNet and CFA-Net all write `reduce='none'`; `reduce` is the removed
        legacy kwarg, and any truthy string maps to `reduction='mean'`. So their `bce` is
        already a scalar, `(weit*bce).sum()/weit.sum()` collapses to that same scalar, and
        the boundary weight map never touches the BCE term. Ours does — a different loss
        surface, not a stricter version of theirs.

        `iou_smooth` is their +1 against our +1e-5. On a 352^2 map with a small polyp the
        weighted intersection is O(1e2-1e3), so +1 is a visible pull of the ratio toward 1,
        i.e. toward a smaller loss on exactly the smallest targets (ETIS, ColonDB).
        """
        mask = mask.view_as(pred).float()
        pred = pred.float()
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        if self.hparams.bce_unweighted:
            wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        else:
            bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
            wbce = (weit * bce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)
        eps = self.hparams.iou_smooth
        ps = torch.sigmoid(pred)
        inter = ((ps * mask) * weit).sum(dim=(2, 3))
        union = ((ps + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + eps) / (union - inter + eps)
        return (wbce + wiou).mean()

    def _tversky_loss(self, pred, mask):
        """Tversky in BAMPolyp's exact algebra (bampolyp.py:275):
        TI = TP / (TP + alpha*FP + beta*FN), so **alpha weights false positives**.

        The Tversky term this file used to carry was (0.3, 0.7) — the recall-biased
        direction, which pushes toward over-segmentation; that is why it was removed (see
        _seg_loss_pranet's docstring). BAMPolyp runs the *opposite* asymmetry, (0.7, 0.3),
        on precisely the two low-contrast datasets we lose on, and precision bias is the
        direction our own failure mode calls for.

        Caveat worth keeping in view: this is an *area* penalty. A distant false-positive
        blob of a few hundred pixels barely moves it while adding tens of pixels to HD95,
        so do not expect this term alone to repair the boundary metric.
        """
        mask = mask.view_as(pred).float()
        p = torch.sigmoid(pred.float())
        dims = (2, 3)
        tp = (p * mask).sum(dim=dims)
        fp = (p * (1 - mask)).sum(dim=dims)
        fn = ((1 - p) * mask).sum(dim=dims)
        alpha, beta = self.hparams.tversky_alpha, self.hparams.tversky_beta
        ti = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return (1.0 - ti).mean()

    def _boundary_dice_loss(self, pred, mask):
        """1 - Dice over the inner boundary band, BAMPolyp's BoundaryLoss (bampolyp.py:298).

        The band is `m - erode(m)` with erosion written as `1 - maxpool(1 - m, k=3)`, which
        is differentiable and runs on the soft probability map directly — unlike the Sobel
        edge target in _fcc_loss, it needs no precomputed edge maps and no threshold.
        """
        mask = mask.view_as(pred).float()
        p = torch.sigmoid(pred.float())

        def band(m):
            return m - (1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1))

        b_p, b_g = band(p), band(mask)
        inter = (b_p * b_g).sum(dim=(2, 3))
        union = b_p.sum(dim=(2, 3)) + b_g.sum(dim=(2, 3))
        return (1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)).mean()

    @staticmethod
    def _to_gt_res(p, y):
        p = torch.clamp(p.float(), min=-100.0, max=100.0)
        if p.shape[-2:] == y.shape[-2:]:
            return p
        return F.interpolate(p, size=y.shape[-2:], mode='bilinear', align_corners=False)

    def _edge_map(self, x, threshold=None):
        # x: (B,1,H,W) float in [0,1] — binary mask or sigmoid prob map
        kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        # 1e-4, not 1e-8: under fp16 autocast (conv2d always casts to fp16 regardless of
        # input dtype), 1e-8 underflows to exactly 0.0, and sqrt's backward at 0 is inf/nan.
        mag = torch.sqrt(gx * gx + gy * gy + 1e-4)
        amax = mag.amax(dim=(2, 3), keepdim=True)
        # A uniform input (all-background mask, or a small polyp erased by nearest-neighbor
        # downsampling at the 0.75x multi-scale rate) has amax at the sqrt(1e-4) noise floor —
        # dividing by itself would turn "no edges anywhere" into "edges everywhere".
        has_signal = amax > 0.02  # comfortably above the sqrt(1e-4)=0.01 noise floor
        mag = torch.where(has_signal, mag / (amax + 1e-8), torch.zeros_like(mag))
        if threshold is not None:
            mag = (mag > threshold).float()
        return mag

    def _fcc_loss(self, preds, y):
        # preds = (S1, S2, S3, S_g, edge_logit[, S_full][, embedding]). Seg heads: equal-weight
        # wBCE+wIoU (FCC-Net recipe). Edge head: BCE-with-logits vs the GT's Sobel boundary.
        # S_full is present only under --fullres-head and is what net.py predicts
        # with at inference, so it carries weight 1.0 and takes over `out` below.
        embedding = None
        if self.contrast_loss is not None:
            preds, embedding = preds[:-1], preds[-1]
        S1, S2, S3, S_g, edge_logit = preds[:5]
        S_full = preds[5] if len(preds) == 6 else None
        heads = [self._to_gt_res(p, y) for p in (S1, S2, S3, S_g)]
        if self.hparams.ds_mutation:
            seg = sum(self._seg_loss_pranet(sum(heads[i] for i in s), y)
                      for s in _nonempty_subsets(len(heads)))
            # EMCAD sums all 15 subset losses unnormalised (trainer.py:139-148), but under
            # SGD at lr=0.05. We run AdamW at 1e-4 behind a *value* gradient clip of 0.5,
            # which is not scale-invariant, so a 15/4 larger loss would silently retune the
            # effective step size and confound the ablation. Rescaling to the 4-head sum's
            # magnitude keeps --ds-mutation a pure change of supervision structure.
            seg = seg * (len(heads) / _N_SUBSETS[len(heads)])
        else:
            seg = 0.0
            for p in heads:
                seg = seg + self._seg_loss_pranet(p, y)

        full = None
        if S_full is not None:
            full = self._to_gt_res(S_full, y)
            seg = seg + self._seg_loss_pranet(full, y)

        # Optional precision-biased terms, applied to whatever the model actually predicts
        # with at inference (net.py: S_full under --fullres-head, else the summed heads under
        # --ds-mutation, else S1) so the thing being optimised and the thing being scored are
        # the same tensor.
        if self.hparams.tversky_weight or self.hparams.boundary_dice_weight or embedding is not None:
            out = full if full is not None else (
                sum(heads) if self.hparams.ds_mutation else heads[0])
            if embedding is not None:
                # `out` is what the model predicts with, so the hard/easy anchor split is taken
                # against the same tensor that is being scored.
                seg = seg + self.hparams.contrast_weight * self.contrast_loss(embedding, y, out)
            if self.hparams.tversky_weight:
                seg = seg + self.hparams.tversky_weight * self._tversky_loss(out, y)
            if self.hparams.boundary_dice_weight:
                seg = seg + self.hparams.boundary_dice_weight * self._boundary_dice_loss(out, y)

        y_1 = y.float().view(y.shape[0], 1, *y.shape[-2:])
        edge_gt = self._edge_map(y_1, threshold=0.3)
        # Dilate to match the reference: FCC-Net thickens its edge GT with a 5x5 kernel at
        # input resolution before use (FCC-Net-main/utils_2/data_val.py:132,153). Our port
        # matched their lambda=3.0 but not the target. The edge head predicts at f1
        # resolution and is upsampled 8x, so the finest stroke it can express is ~8 input
        # px; against a 1-2px Sobel edge that is ~2% positive, the loss-minimising answer is
        # ~0 everywhere — and CCM's `s*edge + s` then degenerates to `s`, making the whole
        # edge pathway (and the region calibration sitting on it) inert.
        k = edge_dilate_kernel(edge_gt.shape[-2], self.hparams.img_size)
        edge_gt = F.max_pool2d(edge_gt, kernel_size=k, stride=1, padding=k // 2)
        # The decoder hands back the edge logit (fcc_modules.py: `up(edge_logit)`), so this is
        # the numerically stable log-sum-exp form with a gradient bounded by 1. Same value as
        # BCE-on-probability, without the -1/p blow-up that NaN'd W1 at epoch 39.
        edge_loss = F.binary_cross_entropy_with_logits(edge_logit.float(), edge_gt)
        return seg + self.hparams.fcc_edge_weight * edge_loss

    def _ds_loss(self, preds, y):
        # Return order: (final, finest aux, mid aux, coarsest aux); finer heads weigh more.
        if isinstance(preds, (tuple, list)):
            weights = [1.0, 0.5, 0.25, 0.125]
            return sum(w * self._seg_loss_pranet(self._to_gt_res(p, y), y)
                       for w, p in zip(weights, preds))
        return self._seg_loss_pranet(self._to_gt_res(preds, y), y)

    def _dsra_bg_loss(self, bg_logits, y):
        # bg_logits: (bg3, bg2, bg1) — mirrors ds3/ds2/ds1's per-stage weighting.
        y = y.float()
        return sum(w * F.binary_cross_entropy_with_logits(self._to_gt_res(p, y), 1 - y)
                   for w, p in zip([0.125, 0.25, 0.5], bg_logits))

    def _compute_loss(self, preds, y):
        if self.hparams.model_name == 'polyp_pvt':
            # Both heads carry equal weight and both are used at test time (their eval sums
            # the logits), so neither is auxiliary — matches Train.py:84-86.
            return sum(self._seg_loss_pranet(self._to_gt_res(p, y), y) for p in preds)

        if self.hparams.use_fcc_decoder:
            return self._fcc_loss(preds, y)

        bg_logits = None
        if self.hparams.use_dsra and self.hparams.use_boundary_attn:
            preds, bg_logits = preds[:4], preds[5:8]
        elif self.hparams.use_dsra:
            preds, bg_logits = preds[:4], preds[4:7]
        elif self.hparams.use_boundary_attn:
            preds = preds[:4]

        loss = self._ds_loss(preds, y)
        if bg_logits is not None:
            loss = loss + self._dsra_bg_loss(bg_logits, y)
        return loss

    # ---------------------------------------------------------------- steps

    def _scales(self, base_size):
        if self.hparams.multi_scale:
            # PraNet/Polyp-PVT convention: 0.75x / 1x / 1.25x, snapped to a multiple of 32
            # (required by the stride-32 encoder).
            sizes = [int(round(base_size * r / 32) * 32) for r in (0.75, 1.0, 1.25)]
        else:
            sizes = [base_size]
        # The snapping above is load-bearing, not cosmetic: the FCC decoder's CCM cascade
        # halves S_g with a flooring interpolate while the SCF branch inherits the encoder's
        # own floored strides, so an unsnapped 0.75*352=264 makes them disagree by one pixel
        # and CCM fails on `x_up * x_down`. The --no-multi-scale path does no snapping at all,
        # so validate here rather than trusting the dataloader's output size.
        bad = [s for s in sizes if s % 32]
        if bad:
            raise ValueError(
                f"training sizes must be divisible by 32 (stride-32 encoder + FCC decoder); "
                f"got {bad} from base_size={base_size}"
            )
        return sizes

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        opt = self.optimizers()
        accum = self.hparams.accumulate_grad_batches
        sizes = self._scales(x.shape[-1])

        losses = []
        for size in sizes:
            if size != x.shape[-1]:
                x_s = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
                # Deliberate deviation from Polyp-PVT's Train.py, which upsamples the mask
                # bilinearly along with the image: that yields fractional GT values, and every
                # loss downstream (wBCE/wIoU, the Sobel edge map) assumes a binary target.
                # Nearest keeps it binary. --ms-mask-interp bilinear restores the reference
                # form, so the deviation is measurable rather than merely documented.
                y_s = (F.interpolate(y, size=(size, size), mode='nearest')
                       if self.hparams.ms_mask_interp == 'nearest'
                       else F.interpolate(y, size=(size, size), mode='bilinear',
                                          align_corners=False))
            else:
                x_s, y_s = x, y

            loss = self._compute_loss(self(x_s), y_s)
            losses.append(loss.detach())

            if self.hparams.step_per_scale:
                opt.zero_grad()
                self.manual_backward(loss)
                self._clip(opt)
                opt.step()
                self._ema_update()
            else:
                self.manual_backward(loss / (len(sizes) * accum))

        if not self.hparams.step_per_scale:
            is_last = (batch_idx + 1) == self.trainer.num_training_batches
            if (batch_idx + 1) % accum == 0 or is_last:
                self._clip(opt)
                opt.step()
                opt.zero_grad()
                self._ema_update()

        mean_loss = torch.stack(losses).mean()
        # Fail loud and immediately. W1 went NaN at epoch 39 and kept running: every later step
        # was skipped, the GradScaler halved to 0, and the epoch-40 ModelCheckpoint overwrote
        # last.ckpt with NaN weights — so the crash destroyed the resume point as well as the
        # run. One sync per batch is a cheap price for dying at the first bad step instead.
        if not torch.isfinite(mean_loss):
            raise RuntimeError(
                f"non-finite training loss at epoch {self.current_epoch}, batch {batch_idx} "
                f"(per-scale: {[float(l) for l in losses]}). Stopping so that last.ckpt keeps "
                f"the last good weights. Do NOT resume from a checkpoint saved after this point."
            )
        self.log("train_loss", mean_loss, prog_bar=False, on_epoch=True, on_step=False)
        self._log_module_telemetry()
        return mean_loss

    def _log_module_telemetry(self):
        """Surface whether each residual branch is actually engaging.

        The -2.0 gate cost a 200-epoch run precisely because nothing reported that it was
        still shut. These four series make that failure visible by epoch 5:
        `ls_*` should grow by >10x within ~10 epochs, and `edge_mean` trending to ~0 means
        the edge pathway is inert and the region calibration is multiplying zero.
        """
        # Both attachment points, tagged. Reading only `bottleneck` made every --mamba-stage h8
        # run log nothing at all: that path leaves `bottleneck` an nn.Identity and puts the scan
        # on `mamba_h8`, so the one series that says whether the scan engaged went missing on
        # exactly the arm it was written to check.
        for tag, net in (('', getattr(self.model, 'bottleneck', None)),
                         ('h8_', getattr(self.model, 'mamba_h8', None)),
                         ('h4_', getattr(self.model, 'mamba_h4', None))):
            for name in ('ls_h', 'ls_w', 'ls_f'):
                ls = getattr(net, name, None)
                if ls is not None:
                    self.log(f"tel/{tag}{name}", ls.gamma.detach().abs().mean(),
                             on_epoch=True, on_step=False)
            # ls_f gates the fused spatial+spectral pair, so it cannot attribute the frequency
            # axis. This is the signed per-bin gain itself: pinned at 0 means mag_head never
            # left its zero init and the spectral branch is an exact identity.
            dm = getattr(net, 'last_delta_mag', None)
            if dm is not None:
                self.log(f"tel/{tag}delta_mag", dm, on_epoch=True, on_step=False)
            # Fraction of bins at or past the gain bound. Under --freq-gain clamp that is the
            # share of the spectrum getting zero gradient, so a rising series means the branch
            # is freezing, not learning — the failure that sank A0 and was only visible
            # because delta_mag is recorded pre-bound.
            ds = getattr(net, 'last_delta_sat', None)
            if ds is not None:
                self.log(f"tel/{tag}delta_sat", ds, on_epoch=True, on_step=False)
        dec = getattr(self.model, 'fcc_decoder', None)
        if dec is None:
            return
        if hasattr(dec, 'last_edge_mean'):
            self.log("tel/edge_mean", dec.last_edge_mean, on_epoch=True, on_step=False)
        calib = getattr(dec, 'region_calib', None)
        if calib is not None and hasattr(calib, 'last_band_frac'):
            self.log("tel/band_frac", calib.last_band_frac, on_epoch=True, on_step=False)
            self.log("tel/abs_delta", calib.last_abs_delta, on_epoch=True, on_step=False)
            # Zero-init'd output conv: if this stays at 0 the prototype branch never engaged
            # and the --fcc-region-proto row is measuring nothing.
            proto = getattr(calib, 'proto', None)
            if proto is not None and proto.last_ctx_abs is not None:
                self.log("tel/proto_ctx", proto.last_ctx_abs, on_epoch=True, on_step=False)
        # RFP: `rfp_gamma` stuck at its 1e-4 init means the module never engaged and the
        # ablation row is measuring nothing; `rfp_w*` are the mean mixture weights over the
        # dilation rates — if they stay uniform, the prior is being ignored.
        mix = getattr(dec, 'rfp_mix', None)
        if mix is not None:
            self.log("tel/rfp_gamma", mix.gamma.detach().abs().mean(), on_epoch=True, on_step=False)
            if mix.last_weights is not None:
                for rate, w in zip(mix.rates, mix.last_weights):
                    self.log(f"tel/rfp_w{rate}", w, on_epoch=True, on_step=False)

    def _clip(self, opt):
        if self.hparams.grad_clip_val and self.hparams.grad_clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=self.hparams.grad_clip_val,
                                gradient_clip_algorithm=self.hparams.grad_clip_algorithm)

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        logit = self(x)
        if isinstance(logit, (tuple, list)):
            logit = logit[0]
        prob = torch.sigmoid(self._to_gt_res(logit, y))
        pred = (prob > 0.5).float()
        y = y.view_as(pred).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        # Both empty => perfect agreement, score 1.0 (matches the benchmark convention).
        dice = torch.where(denom > 0, (2 * inter) / denom.clamp(min=1e-8), torch.ones_like(denom))
        self.log("val_dice", dice.mean(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def configure_optimizers(self):
        # Uniform LR by default. The previous 0.1x encoder throttle was inherited from a
        # fine-tuning heuristic, but every PraNet-family reference — including Polyp-PVT,
        # which beats us with the same PVTv2-B2 encoder — trains the encoder at the full
        # rate (Train.py:198-201 passes model.parameters() as one group).
        scale = self.hparams.encoder_lr_scale
        lr = self.hparams.learning_rate
        if scale == 1.0:
            groups = [{'params': list(self.parameters()), 'lr': lr}]
        else:
            enc = list(self.model.backbone.parameters()) if self.hparams.model_name == 'polyp_pvt' \
                else list(self.model.encoder.parameters())
            enc_ids = {id(p) for p in enc}
            other = [p for p in self.parameters() if id(p) not in enc_ids]
            groups = [{'params': enc, 'lr': lr * scale}, {'params': other, 'lr': lr}]

        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=self.hparams.weight_decay)

        warmup = self.hparams.warmup_epochs
        total = self.hparams.max_epochs

        cycle = self.hparams.restart_cycle_len

        def lr_lambda(epoch):
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / warmup
            if self.hparams.lr_schedule == 'const':
                return 1.0
            if self.hparams.lr_schedule == 'cosine-restarts':
                # Constant restart amplitude (Huang et al., Snapshot Ensembles 2017): each
                # cycle ends annealed, and successive cycles land in different basins, which
                # is exactly the diversity src/soup.py needs. Plain cosine collapses into one
                # basin — measured here as an EMA-minus-raw gap of +0.0000 by epoch 80.
                # Anchored to the absolute epoch, not to epoch-warmup: ModelCheckpoint fires
                # at current_epoch 19/39/59..., so only an unshifted cycle puts a saved
                # checkpoint on each cycle's annealed minimum instead of 5 epochs past it.
                progress = (epoch % cycle) / cycle
            else:
                progress = (epoch - warmup) / max(1, total - warmup)
            # eta_min/base_lr = 1e-6/1e-4 = 0.01, so the LR never fully stalls.
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # interval 'epoch' is nominal: manual optimization means Lightning does not drive
        # the scheduler, on_train_epoch_end does.
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--model", choices=MODELS, default='ours',
                   help="'ours' = SpectraMambaNet; 'polyp_pvt' = the reference baseline")
    p.add_argument("--recipe", choices=('ours', 'polyppvt'), default=None,
                   help="Preset that overrides the training hyperparameters below. "
                        "'polyppvt' reproduces Polyp-PVT's published recipe exactly "
                        "(defaults to --recipe polyppvt when --model polyp_pvt).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0,
                   help="Seed for the held-out val split ONLY, deliberately independent of "
                        "--seed. Multi-seed runs exist to measure init/ordering variance; if "
                        "the split moved with them, that variance would be confounded with a "
                        "different train/val partition and no significance test on the result "
                        "would mean anything.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation (step_per_scale must be off)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--aug", choices=('none', 'light', 'endo', 'heavy'), default=None,
                   help="Augmentation preset (see data/datamodule.py:build_transform). Passing it "
                        "explicitly SURVIVES --recipe polyppvt — the preset only fills it in when "
                        "it was omitted. It used to be overwritten unconditionally, so every arm "
                        "ever run trained with augmentation off. Omitted: 'none' under --recipe "
                        "polyppvt (Polyp-PVT's published setting), 'light' otherwise.")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Held-out slice of the 1450 training images used for checkpoint selection. "
                        "0 disables it, which also disables honest checkpoint selection.")
    p.add_argument("--lr-schedule", choices=('const', 'cosine', 'cosine-restarts'),
                   default='cosine',
                   help="'cosine-restarts' anneals over --restart-cycle-len epochs and jumps "
                        "back to the base LR, so every cycle boundary is an annealed point in "
                        "a different basin. Feeds src/soup.py: plain cosine ends in one basin "
                        "and the checkpoints are near-identical, while constant LR gives "
                        "diverse but noisy endpoints. Restarts give both.")
    p.add_argument("--restart-cycle-len", type=int, default=20,
                   help="Epochs per cosine cycle under 'cosine-restarts'. Leave at 20: "
                        "ModelCheckpoint saves every 20 epochs under --val-fraction 0, so a "
                        "20-epoch cycle drops a checkpoint exactly at each cycle minimum.")
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--encoder-lr-scale", type=float, default=None,
                   help="LR multiplier on the encoder parameter group; 1.0 (the default, and "
                        "the PraNet-family convention) trains it at the full rate. Passing it "
                        "explicitly SURVIVES --recipe polyppvt — the preset only fills it in "
                        "when it was omitted. SAM-Mamba (Sec. 3.2) and HAT-SAM3 both freeze "
                        "the pretrained encoder outright and train adapters; 0.1 is the "
                        "throttled version of that mechanism, and it is the only lever the "
                        "A8 arm has.")
    p.add_argument("--grad-clip-val", type=float, default=0.5)
    p.add_argument("--grad-clip-algorithm", choices=('value', 'norm'), default='value')
    p.add_argument("--no-multi-scale", action="store_false", dest="multi_scale", default=True)
    p.add_argument("--no-step-per-scale", action="store_false", dest="step_per_scale", default=True)
    p.add_argument("--zeroshot-every", type=int, default=20,
                   help="Epoch cadence for the zero-shot monitoring print (0 = off). "
                        "Monitoring only — never drives checkpoint selection.")
    p.add_argument("--zeroshot-history", default="zeroshot_history.json",
                   help="Where the per-epoch monitoring scores are appended. Needed to recover "
                        "the best-epoch-on-test number Polyp-PVT publishes, for comparison "
                        "against our val-selected one. '' to disable.")
    p.add_argument("--select-on-test", action="store_true",
                   help="Also keep a rolling best-on-test.ckpt, chosen by whichever score "
                        "--select-on names. This is the only way a --val-fraction 0 run has a "
                        "checkpoint at the epoch it peaked on. Under the default "
                        "--select-on 5split the result is a max-over-epochs-on-test statistic "
                        "and must be labelled as such; under --select-on indomain it is not, "
                        "because ZS-3 never enters the selection signal.")
    p.add_argument("--select-on", choices=["5split", "indomain"], default="5split",
                   help="Which score --select-on-test maximises. 'indomain' (Kvasir + "
                        "CVC-ClinicDB only) is the published rule: Polyp-PVT logs the five "
                        "benchmark splits but selects on a separate in-domain 'test' folder "
                        "(Train.py:116-121), so CVC-300/ColonDB/ETIS never touch selection and "
                        "the reported ZS-3 stays blind. '5split' is our historical rule and is "
                        "strictly more test-peeked than the papers we compare to; it is kept "
                        "only so archived runs remain comparable.")
    p.add_argument("--select-weights", choices=["auto", "raw"], default="auto",
                   help="Which weights --select-on-test scores. 'auto' prefers the EMA shadow "
                        "when --ema-decay is on. Pass 'raw' when the paper's headline is the "
                        "raw column (the 2026-08-08 decision): otherwise best-on-test.ckpt is "
                        "the peak of the EMA curve and its raw weights are off-peak.")
    p.add_argument("--train-root", default=os.environ.get("TRAIN_DATA_ROOT", "/content/TrainDataset"),
                   help="Folder holding the 1450 training pairs as image/ and masks/.")
    p.add_argument("--test-root", default=os.environ.get("TEST_DATA_ROOT", "/content/TestDataset"),
                   help="Folder holding the five test splits, each with images/ and masks/.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--matmul-precision", choices=["medium", "high", "highest"], default="medium",
                   help="torch.set_float32_matmul_precision. 'medium' is the module-level "
                        "default and keeps fp32 matmuls at reduced internal precision on "
                        "Ampere+, so --precision 32 alone is NOT an fp32 probe. Every model we "
                        "reproduce (Polyp-PVT, CTNet, HGM) runs plain fp32 with no autocast; "
                        "pass 'highest' together with --precision 32 to actually match them.")
    p.add_argument("--fresh", action="store_true", help="Delete all checkpoints and train from scratch")

    p.add_argument("--mamba-bidirectional", action="store_true",
                   help="Pair every tri-axial scan with a reversed scan and merge them "
                        "(HGM lib/Decoder.py:73-161). Without it each scan is causal and the "
                        "first token of a row never sees the last. Cost scales as "
                        "3*(6.25*d^2 + 110*d) at the attached stage: ~+5.1 M at "
                        "--mamba-stage bottleneck (d=512), but only ~+0.35 M at h8 (d=128) "
                        "and ~+0.10 M at h4 (d=64). The +5.1 M figure quoted here previously "
                        "was the bottleneck number and priced this flag out of every h8 run.")
    p.add_argument("--ds-mutation", action="store_true",
                   help="Mutation supervision (EMCAD/MERIT/MIST): apply the seg loss to every "
                        "one of the 15 non-empty summed subsets of the four FCC seg heads, and "
                        "predict with their sum at inference. Without it the heads are trained "
                        "independently and only S1 is used, while the Polyp-PVT baseline we are "
                        "measured against already predicts with p1+p2.")
    p.add_argument("--no-mamba-bottleneck", action="store_false", dest="use_mamba_bottleneck",
                   default=True,
                   help="Replace the tri-axial Mamba with nn.Identity, keeping the FCC decoder. "
                        "This is the attribution ablation: without it we cannot say what the "
                        "module contributes, and the paper cannot ship.")
    p.add_argument("--mamba-stage", type=str, default='bottleneck',
                   choices=['bottleneck', 'h8', 'both', 'h4', 'h4h8'],
                   help="Where the tri-axial Mamba attaches. 'bottleneck' is f3 at H/32 = "
                        "11x11 at 352, i.e. 11-token axial scans. 'h8' moves it to f1 "
                        "(44x44 = 1936 tokens, 128 ch), which is where HGM puts its scans and "
                        "which is also the feature feeding the FCM edge stream. 'both' keeps "
                        "the deep scan as well. 'h4' scans f0 (88x88 = 7744 tokens, 64 ch), "
                        "the only skip reaching S1, the head we predict with; a 15-31 px "
                        "ETIS polyp is 5.4 cells there against 2.7 at H/8. 'h4h8' scans both "
                        "fine stages and drops the deep one — HGM's actual placement once its "
                        "inert window partitioning is stripped. h4 adds ~+0.10 M params at "
                        "d=64 but is the most expensive in wall clock: 7744-token sequences "
                        "parallelise far worse than 88 rows of 88.")
    p.add_argument("--mamba-ls-init", type=float, default=1e-4,
                   help="Initial value of every tri-axial LayerScale gamma. 1e-4 is CaiT's "
                        "value for 24-deep stacks; measured here it only reaches ~1e-2 after "
                        "107 epochs, so the scan contributes ~1%% of the residual while the "
                        "analytic MAC floor charges it 2.41 G (12%% of the model). 0.1 puts "
                        "the branch in the residual stream from step 0 at identical cost. If "
                        "that does not help, the honest move is to drop mamba_h4 and reclaim "
                        "~1.27 G rather than keep paying for an inert branch.")
    p.add_argument("--no-freq-axis", action="store_false", dest="mamba_freq_axis",
                   default=True,
                   help="Drop the frequency-band scan from every tri-axial block, leaving a "
                        "bi-axial H+W spatial scan (no norm_f/mamba_f/mag_head/spectral_fuse "
                        "parameters are created). This is the tri-vs-bi-axial ablation: the "
                        "delta it isolates is the frequency axis itself — the specific claim "
                        "over bi-axial spatial scanning (BAMPolyp) and tri-orientated "
                        "*spatial* scanning (SegMamba).")
    p.add_argument("--freq-gain", type=str, default='clamp', choices=['clamp', 'tanh'],
                   help="How the spectral branch's signed per-bin log-gain is bounded. "
                        "'clamp' is the original hard bound and is what ablation arm A0 ran: "
                        "its h4 telemetry reported a mean pre-bound |gain| of 3.876 against a "
                        "bound of 3.0, i.e. most bins sat outside the clamp receiving zero "
                        "gradient, frozen at a x20 amplify or a /20 suppress, and A0 came in "
                        "0.0061 ZS-3 *below* the bi-axial --no-freq-axis arm on all five "
                        "splits. 'tanh' reaches the same range smoothly so no bin is ever "
                        "gradient-dead. Watch tel/*delta_sat to tell the two apart.")
    p.add_argument("--freq-gain-scale", type=float, default=3.0,
                   help="Range of the log-gain: the bound for --freq-gain clamp, the "
                        "amplitude for tanh. 3.0 admits gains in [e^-3, e^3] = [0.05, 20], "
                        "which is far more range than spectral shaping needs and is what let "
                        "A0's h4 branch rail. 1.0 gives [0.37, 2.7] and 0.5 gives [0.6, 1.6].")
    p.add_argument("--freq-gain-shared", action="store_true",
                   help="Emit one gain per frequency bin shared across channels "
                        "(mag_head becomes Linear(dim, 1)) instead of one per bin per "
                        "channel. At h4 the per-channel form has 88x45xC independent knobs on "
                        "the deepest pretrained features — enough to memorise the imaging "
                        "fingerprint of the two training centres rather than a transferable "
                        "spectral prior, which is the shape of A0's failure (-0.004 CVC-300, "
                        "-0.001 ColonDB, -0.013 ETIS, worst on the most distant centre). "
                        "Sharing collapses it to a single curve over frequency.")
    p.add_argument("--third-axis", type=str, default='freq', choices=['freq', 'raster'],
                   help="What the block's third branch scans. 'freq' is the contribution: the "
                        "sequence dimension is the rfft2 bin index. 'raster' is its control — "
                        "the identical branch (same norm_f/mamba_f(+rev)/mag_head/"
                        "spectral_fuse/ls_f, so the same parameter count, and the same bounded "
                        "multiplicative gain that is exact identity at zero-init) run over a "
                        "row-major flattening of the spatial map. It separates 'the third axis "
                        "is frequency' from 'there is a third branch', which is the first "
                        "thing a reviewer will ask of a tri-axial claim. The control is if "
                        "anything favoured: H*W tokens is ~2x the H*(W//2+1) rfft2 grid.")
    p.add_argument("--mamba-scan", type=str, default='axial', choices=['axial', 'raster'],
                   help="Token order for the two spatial scans. 'axial' scans each row and "
                        "then each column, so no state ever crosses a row boundary. 'raster' "
                        "flattens the whole map into one sequence, row-major then "
                        "column-major — what HGM does (lib/Decoder.py:442-481, the stack run "
                        "on x and again on x.permute(0,1,3,2)). Identical parameter count "
                        "either way, so the two are directly comparable. Pair with "
                        "--mamba-stage h8: 1936 tokens is the regime a selective scan is for.")
    p.add_argument("--fcc-region-proto", action="store_true",
                   help="Add HGM's BDFM region-prototype re-projection to BoundaryRegionCalib "
                        "(lib/pvt.py:490-527). The calib already builds the same "
                        "{interior, boundary, exterior} split but only emits a scalar logit "
                        "delta from it; this pools a prototype per region and re-projects it "
                        "into the feature map. Zero-init'd output conv, so it is an exact "
                        "identity at step 0.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Decay of an exponential moving average of the weights (0 = off, "
                        "0.999 is the arm on the run list). The zero-shot monitor scores both "
                        "the raw and the EMA weights every epoch and writes both to the "
                        "history JSON. Free at inference, and its 'last' epoch is a "
                        "non-test-peeked number — unlike anything read off best-on-test.ckpt.")
    p.add_argument("--encoder", type=str, default='pvt_v2_b2',
                   choices=['pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5'],
                   help="PVTv2 backbone variant. B2-B5 share embed_dims [64,128,320,512] and "
                        "differ only in depth, so the decoder is unchanged; only params and "
                        "compute move (backbone alone: B2 25.4M, B3 45.2M, B4 62.6M, B5 82.0M). "
                        "B4/B5 are untested here and risk OOM at batch 16 x 3 multi-scale.")
    p.add_argument("--edge-stride", type=int, default=8, choices=[4, 8],
                   help="Stride at which the FCC edge head predicts. The reference head sits on "
                        "x_h_cat at H/8, so its finest expressible stroke is ~8 input px against "
                        "a 1-2 px boundary. 4 refines it once against f0 (H/4) before the 1x1. "
                        "Matters most on ETIS (196 small low-contrast polyps), our worst split.")
    p.add_argument("--hat-prob", type=float, default=0.0,
                   help="Probability of HAT (highlight-aware target perturbation, HAT-SAM3 "
                        "Sec. 4): with this probability, union the GT mask with the specular "
                        "highlights lying in a narrow exterior band around its boundary. "
                        "The paper uses p=0.5. Target-side only; never applied at val or test. "
                        "Their Table 3 ablation: mDice 0.8710->0.8837, exterior FP "
                        "0.1579->0.1524, Boundary F 0.7831->0.7864.")
    p.add_argument("--tversky-weight", type=float, default=0.0,
                   help="Weight of an FP-asymmetric Tversky term added on top of structure_loss. "
                        "BAMPolyp (bampolyp.py:256-300) trains ClinicDB and ColonDB with "
                        "alpha=0.7 / beta=0.3, i.e. penalising false positives 2.3x harder than "
                        "false negatives. Off by default. Note this is an AREA penalty: a small "
                        "distant blob costs little area but a lot of HD95, so it is a weak "
                        "instrument against our actual HD95 failure mode.")
    p.add_argument("--tversky-alpha", type=float, default=0.7,
                   help="Weight on false positives (Salehi convention: TI = TP/(TP+a*FP+b*FN)). "
                        "Our _seg_loss_pranet docstring records that a recall-biased 0.3/0.7 was "
                        "previously removed for causing over-segmentation; 0.7/0.3 is the "
                        "opposite direction.")
    p.add_argument("--tversky-beta", type=float, default=0.3, help="Weight on false negatives.")
    p.add_argument("--boundary-dice-weight", type=float, default=0.0,
                   help="Weight of a Dice on the 1-px soft boundary band, m - erode(m), taken "
                        "from BAMPolyp's BoundaryLoss (bampolyp.py:298). Needs no precomputed "
                        "edge target. Off by default.")
    p.add_argument("--no-fcc-decoder", action="store_false", dest="use_fcc_decoder", default=True,
                   help="Disable the FCC-Net-derived decoder (FCM octave-freq + SCF + CCM) and fall "
                        "back to the legacy spectral decoder chain. FCC is the default.")
    p.add_argument("--fcc-channel", type=int, default=64,
                   help="Width of the FCC decoder. FCC-Net's own trainer runs 32 "
                        "(MyTrain_Val.py:199); 64 is our default. Note this barely controls the "
                        "FCM octave blocks unless --fcc-reduce-octave is also passed — two of "
                        "Octave's four sub-blocks are sized by the encoder stage width alone.")
    p.add_argument("--fcc-reduce-octave", action="store_true",
                   help="Put a 1x1 in front of each FCM octave block so it runs at "
                        "--fcc-channel width instead of the encoder stage's, and drop the "
                        "weight-tied second application of mid1. rfb4_1 alone is 5.05 M params "
                        "on an 11x11 map today; Polyp-PVT's entire decoder on the same encoder "
                        "is 0.26 M. Expect ~-7.0 M params and ~-1.4 GMACs.")
    p.add_argument("--bce-unweighted", action="store_true",
                   help="Drop the boundary weight map from the BCE half of structure_loss. "
                        "This REPRODUCES the reference: PraNet/Polyp-PVT/CTNet/CFA-Net all pass "
                        "the removed legacy kwarg `reduce='none'`, which resolves to "
                        "reduction='mean', so their weighted BCE is unweighted. Ours is "
                        "genuinely weighted, i.e. a different loss surface. One of three "
                        "candidates for the 0.021 Polyp-PVT reproduction gap.")
    p.add_argument("--iou-smooth", type=float, default=1e-5,
                   help="Smoothing constant in the wIoU term. The reference uses 1.0; ours has "
                        "been 1e-5. Matters most where the weighted intersection is smallest, "
                        "i.e. on small polyps (ETIS, ColonDB).")
    p.add_argument("--ms-mask-interp", choices=('nearest', 'bilinear'), default='nearest',
                   help="How the GT mask is resized for the 0.75x/1.25x multi-scale steps. "
                        "'bilinear' is the reference and yields fractional targets; 'nearest' "
                        "keeps the mask binary and is our default.")
    p.add_argument("--img-size", type=int, default=352,
                   help="Base training/eval resolution; must be a multiple of 32. 352 is the "
                        "PraNet-family convention and the resolution every published number in "
                        "our table was produced at. It is also an information bound: ETIS is "
                        "natively 966x1225, so at 352 we discard 3.5x linear detail before the "
                        "encoder sees anything. MACs scale ~quadratically (384 -> x1.19, "
                        "448 -> x1.62), so cost must be re-measured and reported AT the "
                        "resolution used. Flows to the dataloader resize, the multi-scale "
                        "rates, the edge-GT dilation radius and BoundaryRegionCalib's band; "
                        "the checkpoint carries it, so src/test.py picks it up automatically.")
    p.add_argument("--fullres-head", action="store_true",
                   help="Decode S1 to input resolution with two learned stride-2 transposed "
                        "convs plus the f0 skip, and predict with THAT head instead of the "
                        "bilinearly-upsampled S1. The four existing heads stay supervised. "
                        "This removes --ds-mutation's inference-side role (it exists to make "
                        "the 4-head sum a legally predictable quantity) while leaving it a "
                        "valid multi-head regulariser. Our own oracle puts the 88^2 ceiling at "
                        "0.9673 mDice on ETIS against our 0.7901, so judge this on HD95, not "
                        "on Dice.")
    p.add_argument("--fcc-edge-weight", type=float, default=3.0,
                   help="Weight on the edge head's BCE. 3.0 is FCC-Net's value and the largest "
                        "single coefficient in our loss, never swept. Note the decoder learns "
                        "to ignore what it trains: CCM fuses as conv(s*edge + s) and edge_mean "
                        "decays 0.359 -> 0.029 over a run, so by late training the edge map is "
                        "a ~3%% multiplicative perturbation. Lowering it is the first lever to "
                        "try against over-segmentation on the easy splits.")
    p.add_argument("--contrast-weight", type=float, default=0.0,
                   help="Weight of a pixel-level supervised contrastive term on a projection "
                        "of the deepest feature (model/contrast.py, ported from CTNet's "
                        "Contrastive_head.py + pixel_contrast_cross_entropy_loss.py, which it "
                        "trains at 0.1). Pushes background embeddings away from polyp ones, "
                        "targeting the distant-false-positive failure mode: ETIS Dice 0.782 at "
                        "Recall@0.5 0.9104 and HD95 140.18 px is a precision failure, not a "
                        "missed-polyp one. The head is dropped at inference, so this costs "
                        "ZERO params and ZERO MACs in the reported model.")
    p.add_argument("--no-fcc-region-calib", action="store_false", dest="fcc_region_calib", default=True)
    p.add_argument("--fcc-share-bn", action="store_true",
                   help="Restore the reference FCC-Net form in which one ConvBNReLU is reused "
                        "across call sites with different activation statistics (3 in CCM, 2 in "
                        "FCCDecoder). Off by default because the shared BatchNorm running stats "
                        "cover all sites at once and shift the output between train and eval "
                        "modes; kept as a flag so the deviation stays an attributable ablation.")
    p.add_argument("--use-rfp", action="store_true",
                   help="Rank-normalized Frequency Prior on the decoder's high-frequency "
                        "stream (see src/model/rfp.py and src/analysis/arf_audit.py). Off by "
                        "default: it may not be reported until the Polyp-PVT parity gate has "
                        "passed, or the ablation has no reference to be measured against.")
    p.add_argument("--use-boundary-attn", action="store_true")
    p.add_argument("--use-wavelet-decoder", action="store_true")
    p.add_argument("--use-dsra", action="store_true")
    p.add_argument("--fda-aug-prob", type=float, default=0.0,
                   help="Probability of swapping a training image's low-frequency Fourier "
                        "amplitude with another *training* image's (data/datamodule.py:69-92). "
                        "Phase carries structure and is untouched, so only texture, colour and "
                        "illumination are randomised. Donors come from self.image_paths, i.e. "
                        "the 1450-image training pool — no test image is ever read. This is "
                        "the one augmentation whose mechanism is the metric: the zero-shot "
                        "splits differ from training mostly in amplitude.")
    p.add_argument("--fda-beta", type=float, default=0.01,
                   help="Half-width of the swapped low-frequency window as a fraction of the "
                        "spectrum. 0.01 at 352 is a 7x7 corner patch; larger values start "
                        "moving structure, not just style.")
    return p


def apply_recipe(args):
    """Resolve --recipe presets. Explicit flags are not tracked separately, so a preset
    overwrites them — pass --recipe none-equivalent (omit it) to hand-tune."""
    recipe = args.recipe
    if recipe is None:
        recipe = 'polyppvt' if args.model == 'polyp_pvt' else 'ours'

    if recipe == 'polyppvt':
        # Polyp-PVT Train.py as published: AdamW 1e-4 uniform + wd 1e-4, batch 16,
        # trainsize 352, value-clip 0.5, augmentation off, 3 multi-scale rates each taking
        # their own optimizer step, and a constant LR (adjust_lr's decay_epoch=200 never
        # fires inside a 100-epoch run).
        args.lr = 1e-4
        args.weight_decay = 1e-4
        args.batch_size = 16
        args.accum = 1
        if args.aug is None:
            args.aug = 'none'
        args.lr_schedule = 'const'
        args.warmup_epochs = 0
        args.grad_clip_val = 0.5
        args.grad_clip_algorithm = 'value'
        args.multi_scale = True
        args.step_per_scale = True
    args.recipe = recipe
    # The preset fields that are NOT overwritten. --encoder-lr-scale used to be forced to 1.0
    # here, which made it a silent no-op under the only recipe every arm runs — the run config
    # printed the value the user passed while the optimizer used 1.0. --aug had the same defect
    # in a worse form: it was overwritten with 'none' *and* the config printed 'none', so every
    # measured run trained with augmentation off and the log was honest about it while the
    # invocation was not.
    if args.encoder_lr_scale is None:
        args.encoder_lr_scale = 1.0
    if args.aug is None:
        args.aug = 'light'
    return args


if __name__ == "__main__":
    import glob
    import shutil
    import sys

    args = apply_recipe(build_argparser().parse_args())
    pl.seed_everything(args.seed, workers=True)

    # Overrides the module-level 'medium' set at import. Logged, not silent: a precision probe
    # that leaves this at 'medium' measures nothing and reads as if it did.
    torch.set_float32_matmul_precision(args.matmul_precision)
    print(f"[precision] lightning={args.precision} "
          f"float32_matmul={torch.get_float32_matmul_precision()}")

    print(f"[run-config] model={args.model} encoder={args.encoder} "
          f"recipe={args.recipe} seed={args.seed} "
          f"epochs={args.epochs} bs={args.batch_size} accum={args.accum} lr={args.lr} "
          f"wd={args.weight_decay} aug={args.aug} hat_prob={args.hat_prob} "
          f"fda={args.fda_aug_prob}@beta{args.fda_beta} "
          f"val_fraction={args.val_fraction} "
          f"split_seed={args.split_seed} "
          f"lr_schedule={args.lr_schedule} warmup={args.warmup_epochs} "
          f"restart_cycle_len={args.restart_cycle_len} "
          f"encoder_lr_scale={args.encoder_lr_scale} "
          f"clip={args.grad_clip_algorithm}@{args.grad_clip_val} "
          f"multi_scale={args.multi_scale} step_per_scale={args.step_per_scale} "
          f"use_fcc_decoder={args.use_fcc_decoder} fcc_region_calib={args.fcc_region_calib} "
          f"fcc_edge_weight={args.fcc_edge_weight} fcc_share_bn={args.fcc_share_bn} "
          f"fcc_channel={args.fcc_channel} fcc_reduce_octave={args.fcc_reduce_octave} "
          f"bce_unweighted={args.bce_unweighted} iou_smooth={args.iou_smooth} "
          f"ms_mask_interp={args.ms_mask_interp} img_size={args.img_size} "
          f"fullres_head={args.fullres_head} "
          f"select_on_test={args.select_on_test} select_on={args.select_on} "
          f"select_weights={args.select_weights} "
          f"use_rfp={args.use_rfp} edge_stride={args.edge_stride} "
          f"mamba_stage={args.mamba_stage} mamba_scan={args.mamba_scan} "
          f"mamba_freq_axis={args.mamba_freq_axis} mamba_ls_init={args.mamba_ls_init} "
          f"freq_gain={args.freq_gain} freq_gain_scale={args.freq_gain_scale} "
          f"freq_gain_shared={args.freq_gain_shared} third_axis={args.third_axis} "
          f"fcc_region_proto={args.fcc_region_proto} ema_decay={args.ema_decay} "
          f"precision={args.precision} matmul_precision={args.matmul_precision} "
          f"use_mamba_bottleneck={args.use_mamba_bottleneck} "
          f"ds_mutation={args.ds_mutation} mamba_bidirectional={args.mamba_bidirectional} "
          f"tversky={args.tversky_weight}@({args.tversky_alpha},{args.tversky_beta}) "
          f"boundary_dice={args.boundary_dice_weight} "
          f"contrast_weight={args.contrast_weight}")

    # SpectraMambaNet.__init__ returns as soon as the FCC decoder is built, so every module
    # constructed after that point — BoundaryAttentionModule, the wavelet branch, DSRA, the
    # deep-supervision heads — is never instantiated when use_fcc_decoder is on (the default).
    # These flags were therefore silent no-ops: the run trained the identical network and
    # logged a config claiming otherwise. Fail at launch instead.
    if args.use_fcc_decoder:
        dead = [name for name, on in (("--use-boundary-attn", args.use_boundary_attn),
                                      ("--use-wavelet-decoder", args.use_wavelet_decoder),
                                      ("--use-dsra", args.use_dsra)) if on]
        if dead:
            print(f"FATAL: {' '.join(dead)} has no effect under the FCC decoder — those modules "
                  "live on the legacy decoder path that SpectraMambaNet.__init__ returns before "
                  "reaching. Add --no-fcc-decoder to actually enable them, or drop the flag(s).")
            sys.exit(1)

    # Same failure mode, other direction: RFP attaches to the FCC decoder's high-frequency
    # stream, and PolypPVT has no such stream at all.
    if args.use_rfp and (not args.use_fcc_decoder or args.model != 'ours'):
        print("FATAL: --use-rfp attaches to the FCC decoder's high-frequency stream "
              "(fcc_modules.py::FCCDecoder). It requires --model ours with the FCC decoder "
              f"enabled; got --model {args.model} use_fcc_decoder={args.use_fcc_decoder}.")
        sys.exit(1)

    # Third instance of the same failure mode: both extra loss terms are read only inside
    # _fcc_loss, so on the legacy or PolypPVT loss path they would be silently dropped.
    if (args.tversky_weight or args.boundary_dice_weight
            or args.contrast_weight) and not (
            args.use_fcc_decoder and args.model == 'ours'):
        print("FATAL: --tversky-weight/--boundary-dice-weight/--contrast-weight are applied "
              f"in _fcc_loss only. Got --model {args.model} "
              f"use_fcc_decoder={args.use_fcc_decoder}; the terms would be silently ignored.")
        sys.exit(1)

    if args.contrast_weight < 0:
        print(f"FATAL: --contrast-weight must be >= 0, got {args.contrast_weight}. The head is "
              "built only when it is strictly positive, so a negative value would both flip "
              "the loss and silently skip the projection head.")
        sys.exit(1)

    if (args.fcc_channel != 64 or args.fcc_reduce_octave or args.fullres_head) and not (
            args.use_fcc_decoder and args.model == 'ours'):
        print(f"FATAL: --fcc-channel/--fcc-reduce-octave/--fullres-head size or extend the "
              f"FCC decoder; got --model {args.model} "
              f"use_fcc_decoder={args.use_fcc_decoder}, which has none.")
        sys.exit(1)

    if args.edge_stride != 8 and not (args.use_fcc_decoder and args.model == 'ours'):
        print(f"FATAL: --edge-stride {args.edge_stride} only affects the FCC decoder's edge "
              f"head; got --model {args.model} use_fcc_decoder={args.use_fcc_decoder}.")
        sys.exit(1)

    if args.mamba_scan != 'axial' and not (args.model == 'ours' and args.use_mamba_bottleneck):
        print(f"FATAL: --mamba-scan {args.mamba_scan} reorders the tri-axial scan's tokens; "
              f"got --model {args.model} use_mamba_bottleneck={args.use_mamba_bottleneck}, "
              "which has no scan to reorder.")
        sys.exit(1)

    if args.mamba_ls_init != 1e-4 and not (args.model == 'ours' and args.use_mamba_bottleneck):
        print(f"FATAL: --mamba-ls-init {args.mamba_ls_init} scales the tri-axial block's "
              f"residual; got --model {args.model} "
              f"use_mamba_bottleneck={args.use_mamba_bottleneck}, which has no block to scale.")
        sys.exit(1)

    if not args.mamba_freq_axis and not (args.model == 'ours' and args.use_mamba_bottleneck):
        print(f"FATAL: --no-freq-axis removes the frequency scan from the tri-axial block; "
              f"got --model {args.model} use_mamba_bottleneck={args.use_mamba_bottleneck}, "
              "which has no block to remove it from.")
        sys.exit(1)

    if args.fcc_region_proto and not (args.use_fcc_decoder and args.model == 'ours'
                                      and args.fcc_region_calib):
        print("FATAL: --fcc-region-proto extends BoundaryRegionCalib. Needs --model ours "
              "with the FCC decoder and the region calib both enabled; got "
              f"--model {args.model} use_fcc_decoder={args.use_fcc_decoder} "
              f"fcc_region_calib={args.fcc_region_calib}.")
        sys.exit(1)

    if args.mamba_stage != 'bottleneck' and not (args.use_fcc_decoder and args.model == 'ours'):
        print(f"FATAL: --mamba-stage {args.mamba_stage} places a scan on f1 and/or f0, which "
              f"only the FCC decoder path consumes; got --model {args.model} "
              f"use_fcc_decoder={args.use_fcc_decoder}.")
        sys.exit(1)

    if args.model == 'ours':
        try:
            import mamba_ssm  # noqa: F401
        except ImportError:
            print("FATAL: mamba_ssm is not installed. model.net silently falls back to a")
            print("Linear+GELU stub for CPU-only smoke tests, which would let a real training")
            print("run start on a non-Mamba architecture with no error. Install mamba_ssm")
            print("(pip install mamba-ssm) before training.")
            sys.exit(1)

    datamodule = PolypDataModule(
        data_dir=args.train_root,
        batch_size=args.batch_size, fda_prob=args.fda_aug_prob, fda_beta=args.fda_beta,
        aug=args.aug, val_fraction=args.val_fraction, seed=args.split_seed,
        num_workers=args.num_workers, hat_prob=args.hat_prob, img_size=args.img_size,
    )
    model = SpectraMambaLightning(
        model_name=args.model,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        lr_schedule=args.lr_schedule, restart_cycle_len=args.restart_cycle_len,
        encoder_lr_scale=args.encoder_lr_scale,
        grad_clip_val=args.grad_clip_val, grad_clip_algorithm=args.grad_clip_algorithm,
        multi_scale=args.multi_scale, step_per_scale=args.step_per_scale,
        accumulate_grad_batches=args.accum,
        use_boundary_attn=args.use_boundary_attn, use_wavelet_decoder=args.use_wavelet_decoder,
        use_dsra=args.use_dsra, use_fcc_decoder=args.use_fcc_decoder,
        fcc_edge_weight=args.fcc_edge_weight, fcc_region_calib=args.fcc_region_calib,
        use_rfp=args.use_rfp, fcc_share_bn=args.fcc_share_bn,
        ds_mutation=args.ds_mutation, mamba_bidirectional=args.mamba_bidirectional,
        tversky_weight=args.tversky_weight, tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta, boundary_dice_weight=args.boundary_dice_weight,
        edge_stride=args.edge_stride, mamba_stage=args.mamba_stage,
        use_mamba_bottleneck=args.use_mamba_bottleneck,
        encoder_name=args.encoder,
        mamba_scan=args.mamba_scan, fcc_region_proto=args.fcc_region_proto,
        mamba_freq_axis=args.mamba_freq_axis, mamba_ls_init=args.mamba_ls_init,
        freq_gain=args.freq_gain, freq_gain_scale=args.freq_gain_scale,
        freq_gain_shared=args.freq_gain_shared, third_axis=args.third_axis,
        ema_decay=args.ema_decay,
        fcc_channel=args.fcc_channel, fcc_reduce_octave=args.fcc_reduce_octave,
        bce_unweighted=args.bce_unweighted, iou_smooth=args.iou_smooth,
        ms_mask_interp=args.ms_mask_interp, img_size=args.img_size,
        fullres_head=args.fullres_head, contrast_weight=args.contrast_weight,
    )
    # Printed at launch because --fcc-reduce-octave / --mamba-stage are supposed to move this
    # by millions of parameters, and a flag that silently failed to reach the constructor is
    # this repo's recurring failure mode. Check it against the plan's expected count before
    # leaving the launch cell.
    _parts = {n: sum(p.numel() for p in m.parameters()) / 1e6
              for n, m in (("encoder", getattr(model.model, "encoder", None)),
                           ("fcc_decoder", getattr(model.model, "fcc_decoder", None)))
              if m is not None}
    print(f"[params] total={sum(p.numel() for p in model.model.parameters()) / 1e6:.3f} M "
          + " ".join(f"{n}={v:.3f} M" for n, v in _parts.items()))

    if args.val_fraction > 0:
        # save_top_k=3, not 1: val_dice is computed on ~73 in-domain images and saturates early
        # (it sat in a 0.88-0.91 band from epoch 2 onward in the parity run while zero-shot mDice
        # was still climbing 0.79 -> 0.82). Its argmax over epochs is therefore close to a coin
        # flip among a wide band of epochs, and keeping only the single top-1 file discards the
        # runners-up permanently. Keeping three costs ~475 MB and lets a rerun check how much of
        # the headline number rides on that argmax. audit_accuracy.py --best still resolves to
        # the highest val_dice by parsing it out of the filename, so the default is unchanged.
        checkpoint = ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=3,
                                     save_last=True, filename="best-val-{epoch:03d}-{val_dice:.4f}")
    else:
        print("WARNING: --val-fraction 0 - no held-out split, so there is no blind selection "
              "signal. Report 'last', or pass --select-on-test to reproduce the published "
              "max-over-epochs-on-test rule and label the number accordingly.")
        if args.select_on_test and args.zeroshot_every <= 0:
            sys.exit("--select-on-test needs --zeroshot-every > 0; nothing would be scored.")
        checkpoint = ModelCheckpoint(save_top_k=-1, save_last=True, every_n_epochs=20,
                                     filename="epoch{epoch:03d}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        enable_progress_bar=False,  # Fix Colab newline spam
        num_sanity_val_steps=0,
        # With --val-fraction 0 the datamodule has no val set and val_dataloader() returns
        # None. Lightning 2.x builds the val loop's data regardless and dies in
        # _check_dataloader_iterable ("An invalid dataloader was returned ... Found None").
        # limit_val_batches=0 short-circuits evaluation_loop.setup_data before that check
        # (evaluation_loop.py:164), which is the supported way to say "this run has no
        # validation". Returning [] instead would build an empty CombinedLoader and still
        # run the val-loop machinery every epoch for nothing.
        limit_val_batches=0 if args.val_fraction <= 0 else 1.0,
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            EpochLoggingCallback(),
            AMPScalerTelemetryCallback(),
            ZeroShotMonitorCallback(every_n_epochs=args.zeroshot_every,
                                    history_path=args.zeroshot_history,
                                    select_on_test=args.select_on_test,
                                    select_on=args.select_on,
                                    select_weights=args.select_weights,
                                    test_root=args.test_root),
        ],
        benchmark=True,  # cuDNN auto-tune for speed; set False for strict reproducibility
        deterministic=False,
    )

    if args.fresh:
        if os.path.exists("lightning_logs"):
            print("--fresh: removing lightning_logs/ and starting clean.")
            shutil.rmtree("lightning_logs")
        else:
            print("--fresh: no lightning_logs/ found, already clean.")

    list_of_files = glob.glob('lightning_logs/**/checkpoints/*.ckpt', recursive=True)
    # mtime, not ctime. A crashed Colab runtime loses /content, so the launcher restores the
    # checkpoints with shutil.copytree -> copy2, which carries mtime across but stamps every
    # file's ctime with the restore time. Ordering by ctime then picks whichever file copytree
    # happened to write last. That is not academic here: ModelCheckpoint(every_n_epochs=20)
    # only refreshes last.ckpt on an epoch it actually saved on (PL 2.6 gates it on
    # _last_global_step_saved), so between the 20-epoch marks the newest checkpoint is
    # best-on-test.ckpt from ZeroShotMonitorCallback -- and losing that coin flip silently
    # rewinds the run by up to 19 epochs.
    ckpt_path = max(list_of_files, key=os.path.getmtime) if list_of_files else None
    if ckpt_path:
        print(f"Resuming training from latest checkpoint: {ckpt_path}")

    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    except RuntimeError as e:
        if ckpt_path and ("Missing key(s)" in str(e) or "Unexpected key(s)" in str(e)):
            print("\n" + "=" * 60)
            print("CHECKPOINT ARCHITECTURE MISMATCH")
            print("=" * 60)
            print(f"  Stale checkpoint : {ckpt_path}")
            print("  The model architecture has changed since this checkpoint was saved.")
            print("\n  Run with --fresh to wipe old checkpoints and start clean.")
            print("=" * 60)
            sys.exit(1)
        raise
