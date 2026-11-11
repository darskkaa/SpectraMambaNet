import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image

import os
import random

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif')


def _seed_worker(worker_id):
    # PyTorch reseeds its own (torch) RNG per worker automatically, but not numpy's or
    # Python's global RNG. Under Linux's fork start method (Colab's default), forked
    # workers inherit the parent's exact numpy RNG state, so np.random calls in
    # fda_amplitude_swap (and Albumentations' own numpy-based randomness) are correlated
    # across workers until each has independently consumed enough state to diverge.
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def find_image_mask_pairs(root_dir):
    """Pair image/mask files by basename stem, not directory listing order.

    Sorted-index pairing silently mis-pairs images and masks whenever
    extensions differ or counts diverge (no error, just wrong training/eval
    data). Basename matching is the convention used by PraNet and every
    other polyp-seg reference implementation.
    """
    image_dir = os.path.join(root_dir, 'image')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(root_dir, 'images')

    mask_dir = os.path.join(root_dir, 'mask')
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(root_dir, 'masks')

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError(f"Could not find image/mask directories in {root_dir}")

    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTS)}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXTS)}

    common_stems = sorted(set(image_files) & set(mask_files))
    images_without_masks = sorted(set(image_files) - set(mask_files))
    masks_without_images = sorted(set(mask_files) - set(image_files))

    if images_without_masks or masks_without_images:
        raise ValueError(
            f"Image/mask basename mismatch in {root_dir}: "
            f"{len(images_without_masks)} images without a matching mask "
            f"(e.g. {images_without_masks[:5]}), "
            f"{len(masks_without_images)} masks without a matching image "
            f"(e.g. {masks_without_images[:5]})."
        )
    if not common_stems:
        raise ValueError(f"No matching image/mask basenames found in {root_dir}")

    image_paths = [os.path.join(image_dir, image_files[stem]) for stem in common_stems]
    mask_paths = [os.path.join(mask_dir, mask_files[stem]) for stem in common_stems]
    return image_paths, mask_paths


def fda_amplitude_swap(img, donor, beta=0.01):
    """Swap low-frequency amplitude spectrum with a donor image's, keep phase (structure)
    untouched. FDA-style (Yang & Soatto, CVPR 2020) domain-randomization augmentation —
    targets amplitude-spectrum domain shift (texture/color/lighting) between training data
    and unseen test domains, without altering polyp shape/structure (phase-encoded)."""
    img = img.astype(np.float32)
    donor = donor.astype(np.float32)
    fft_src = np.fft.fft2(img, axes=(0, 1))
    fft_trg = np.fft.fft2(donor, axes=(0, 1))
    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_trg = np.abs(fft_trg)

    h, w = img.shape[:2]
    b = max(1, int(min(h, w) * beta))
    cy, cx = h // 2, w // 2

    amp_src_shift = np.fft.fftshift(amp_src, axes=(0, 1))
    amp_trg_shift = np.fft.fftshift(amp_trg, axes=(0, 1))
    amp_src_shift[cy - b:cy + b, cx - b:cx + b] = amp_trg_shift[cy - b:cy + b, cx - b:cx + b]
    amp_src = np.fft.ifftshift(amp_src_shift, axes=(0, 1))

    fft_mixed = amp_src * np.exp(1j * pha_src)
    out = np.real(np.fft.ifft2(fft_mixed, axes=(0, 1)))
    return np.clip(out, 0, 255).astype(np.uint8)


class CustomPolypDataset(Dataset):
    def __init__(self, root_dir, transform=None, fda_prob=0.0, fda_beta=0.01, indices=None):
        self.transform = transform
        image_paths, mask_paths = find_image_mask_pairs(root_dir)
        if indices is not None:
            image_paths = [image_paths[i] for i in indices]
            mask_paths = [mask_paths[i] for i in indices]
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.fda_prob = fda_prob
        self.fda_beta = fda_beta

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        if self.fda_prob > 0 and np.random.rand() < self.fda_prob:
            donor_idx = np.random.randint(len(self.image_paths))
            donor = Image.open(self.image_paths[donor_idx]).convert("RGB")
            donor = np.array(donor.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR))
            image = fda_amplitude_swap(image, donor, beta=self.fda_beta)

        mask = Image.open(mask_path).convert("L")
        # Safety: force mask to resize to image dimensions to prevent albumentations crash
        mask = mask.resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            
        return {'image': image, 'mask': mask}

HAT_REFERENCE_SIZE = 1008   # HAT-SAM3 trains at 1008^2 and specifies its band radius there
HAT_REFERENCE_RADIUS = 10


def hat_band_radius(size: int) -> int:
    """HAT's 10 px exterior band, rescaled to our input resolution.

    The paper defines the band in pixels at its own 1008^2 training resolution. Copying
    "10 px" to our 352^2 would be a 2.9x wider band in image-relative terms, which is a
    different augmentation, not the paper's. 10 * 352/1008 = 3.49 -> 3.
    """
    return max(1, round(HAT_REFERENCE_RADIUS * size / HAT_REFERENCE_SIZE))


class HighlightAwareTarget(A.DualTransform):
    """HAT (highlight-aware target perturbation), HAT-SAM3 Sec. 4.

    Specular highlights sitting on the polyp rim are annotated inconsistently: sometimes
    inside the mask, sometimes outside. HAT makes the target stochastic exactly where the
    ambiguity lives, so the model is not forced to fit one arbitrary resolution of it.

        V = max_c I_c,  S = (V - min_c I_c) / (V + eps)      # on [0,1] RGB
        H = 1[V > 0.90] * 1[S < 0.25]                        # specular pixels
        E = dilate_r(M) & H & ~M                             # highlights in the exterior band
        M' = M | E    with probability p

    Three details are load-bearing and easy to get wrong:

    - It perturbs the **target only**. The image is returned untouched, and HAT is never
      applied at inference.
    - It must run on the raw [0,1] RGB, i.e. **before** ColorJitter/gamma/blur/noise and
      before Normalize — a jittered V would move both thresholds. Hence build_transform
      inserts it directly after Resize.
    - The paper applies **no morphological cleanup** to H. Opening/closing it would remove
      the single-pixel speculars the augmentation exists to capture.

    The paper does not state eps or the max-pool kernel size; eps=1e-6 and kernel=2r+1
    are the standard readings.
    """

    def __init__(self, p=0.5, radius=3, v_thresh=0.90, s_thresh=0.25):
        super().__init__(p=p)
        self.radius = radius
        self.v_thresh = v_thresh
        self.s_thresh = s_thresh

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        mask = data.get("mask")
        if mask is None:
            return {"extension": None}

        rgb = img.astype(np.float32)
        if img.dtype == np.uint8:
            rgb = rgb / 255.0
        v = rgb.max(axis=2)
        s = (v - rgb.min(axis=2)) / (v + 1e-6)
        highlight = (v > self.v_thresh) & (s < self.s_thresh)

        m = (np.asarray(mask) > 0.5).astype(np.uint8)
        k = 2 * self.radius + 1
        band = cv2.dilate(m, np.ones((k, k), np.uint8)) > 0
        return {"extension": (band & highlight & (m == 0))}

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, extension=None, **params):
        if extension is None or not extension.any():
            return mask
        out = mask.copy()
        out[extension] = 1 if out.dtype == np.uint8 else 1.0
        return out

    def get_transform_init_args_names(self):
        return ("radius", "v_thresh", "s_thresh")


def build_transform(aug: str, size: int = 352, hat_prob: float = 0.0):
    """Augmentation presets.

    'none'  — resize + normalize only. This is Polyp-PVT's published setting
              (`Train.py:158` defaults --augmentation to False) and the setting
              under which it reports 0.808 ColonDB, so it is what the baseline
              reproduction must use.
    'light' — flips + rot90 + mild ColorJitter. Geometric/photometric ops that
              every PraNet-family reference uses when it augments at all.
    'endo'  — 'light' plus gamma/grayscale/blur/noise, one op per axis of the
              scope-to-scope shift the zero-shot splits actually measure.
    'heavy' — the project's historical setting. Kept only so old runs remain
              reproducible; CLAHE and GridDistortion are in here and both are
              suspected net-negative.
    """
    norm = [A.Normalize(), ToTensorV2()]
    resize = [A.Resize(size, size, mask_interpolation=cv2.INTER_NEAREST)]
    # Directly after Resize: HAT thresholds raw [0,1] V and S, so it has to see the image
    # before ColorJitter/CLAHE/blur and before Normalize. The band radius is defined at the
    # working resolution, which is why it goes after Resize rather than before.
    if hat_prob > 0:
        resize = resize + [HighlightAwareTarget(p=hat_prob, radius=hat_band_radius(size))]

    if aug == 'none':
        return A.Compose(resize + norm)
    if aug == 'light':
        return A.Compose(resize + [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        ] + norm)
    if aug == 'endo':
        # HAT-SAM3 (Vis. Comput. 2026) Sec. 3.4 -- the photometric stack the current ZS-3 leader
        # trains with. 'light' carries only the jitter; 'heavy' spends the budget on CLAHE and
        # GridDistortion instead, which the docstring above already calls net-negative.
        return A.Compose(resize + [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.ToGray(p=0.05),
            # albumentations 2.x defaults are far too strong at 352: sigma up to 3.0 and noise
            # sigma up to 0.44 of full range. Scaled to the working resolution.
            A.GaussianBlur(sigma_limit=(0.1, 1.5), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.12), p=0.2),
        ] + norm)
    if aug == 'heavy':
        return A.Compose(resize + [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.CLAHE(p=0.4),
            A.GaussianBlur(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.GridDistortion(p=0.5, mask_interpolation=cv2.INTER_NEAREST),
        ] + norm)
    raise ValueError(f"unknown aug preset: {aug!r} (expected 'none', 'light', 'endo' or 'heavy')")


def _source_of(path):
    """Coarse source label for stratifying the val split.

    PraNet's merged TrainDataset flattens Kvasir-SEG (hash-like stems) and
    CVC-ClinicDB (plain integer stems) into one directory with no provenance
    field. The stem format is the only signal available, and it separates the
    two cleanly. Degrades to a single stratum (i.e. a plain random split) if the
    naming convention ever changes, which is safe rather than silently wrong.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return 'clinicdb' if stem.isdigit() else 'kvasir'


def make_val_split(image_paths, val_fraction, seed):
    """Return (train_indices, val_indices), stratified by source, deterministic in `seed`."""
    by_source = {}
    for i, p in enumerate(image_paths):
        by_source.setdefault(_source_of(p), []).append(i)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for source in sorted(by_source):
        idx = sorted(by_source[source])
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return sorted(train_idx), sorted(val_idx)


class PolypDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="/content/TrainDataset", batch_size=8, fda_prob=0.0,
                 fda_beta=0.01, aug='light', val_fraction=0.05, seed=42, num_workers=4,
                 hat_prob=0.0, img_size=352):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fda_prob = fda_prob
        self.fda_beta = fda_beta
        self.aug = aug
        self.val_fraction = val_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.hat_prob = hat_prob
        self.img_size = img_size
        self.train_transform = build_transform(aug, size=img_size, hat_prob=hat_prob)
        # Val is scored deterministically: no augmentation, ever — and HAT perturbs the
        # target, so scoring against a HAT'd mask would score a different task.
        self.val_transform = build_transform('none', size=img_size)

    def setup(self, stage=None):
        print(f"⏬ Loading WACV Benchmark Train Dataset from {self.data_dir} "
              f"(aug={self.aug}, hat_prob={self.hat_prob}, img_size={self.img_size})...")
        all_images, _ = find_image_mask_pairs(self.data_dir)

        # Held-out in-domain val split, carved from the 1450 TRAIN images only.
        # This exists so ModelCheckpoint has something to select on that is not
        # the benchmark being reported. Selecting on the zero-shot test sets
        # (the previous behaviour) makes every reported number a
        # max-over-checkpoints statistic on its own test set.
        if self.val_fraction > 0:
            train_idx, val_idx = make_val_split(all_images, self.val_fraction, self.seed)
        else:
            train_idx, val_idx = list(range(len(all_images))), []

        self.train_dataset = CustomPolypDataset(
            self.data_dir, transform=self.train_transform,
            fda_prob=self.fda_prob, fda_beta=self.fda_beta, indices=train_idx,
        )
        self.val_dataset = CustomPolypDataset(
            self.data_dir, transform=self.val_transform, indices=val_idx,
        ) if val_idx else None

        n_val = len(self.val_dataset) if self.val_dataset else 0
        n_train = len(self.train_dataset)
        dropped = n_train % self.batch_size  # drop_last=True in train_dataloader
        print(f"✅ Loaded {n_train} train / {n_val} val images "
              f"(val_fraction={self.val_fraction}, split seed={self.seed})")
        # Both deficits belong in the run log: reported numbers come from a model trained on
        # fewer images than the published baselines they are compared against.
        print(f"   effective train images/epoch: {n_train - dropped} of {len(all_images)} "
              f"({n_val} held out, {dropped} dropped by drop_last at batch_size={self.batch_size})")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0,
                          worker_init_fn=_seed_worker, drop_last=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)
