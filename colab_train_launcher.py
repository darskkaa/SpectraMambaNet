#!/usr/bin/env python3
"""
SpectraMambaNet — Complete Colab Training Launcher

IMPORTANT — Mount Drive FIRST (in a separate Colab cell), then run this script:
    from google.colab import drive
    drive.mount('/content/drive')
    !python colab_train_launcher.py

Steps (all idempotent):
  1.  Environment check
  2.  Drive status check (must be pre-mounted from a notebook cell)
  3.  Restore checkpoints from Drive backup (fresh session only)
  4.  Clean workspace (remove empty/stale dataset dirs, keep checkpoints)
  5.  Clone / update repo
  6.  Install dependencies (skip if mamba_ssm already importable; causal-conv1d/mamba-ssm
      wheels are cached to Drive keyed by torch/CUDA/Python version — first run builds from
      source, later runs on the same combo install from cache in seconds)
  7.  Download + extract datasets (PraNet's public zips, or $POLYP_DATA_FOLDER_ID)
  8.  Audit dataset structure (splits, pairing, counts)
  9.  Save datasets to Drive for future sessions
  10. Launch training (auto-resumes from latest checkpoint)
"""

import atexit
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

# ─── constants ────────────────────────────────────────────────────────────────

REPO_URL     = os.environ.get("POLYP_REPO_URL", "https://github.com/darskkaa/SpectraMambaNet.git")
REPO_PATH    = "/content/polp_wacv"
REPO_BRANCH  = os.environ.get("POLYP_REPO_BRANCH", "main")
TRAIN_ROOT   = "/content/TrainDataset"
TEST_ROOT    = "/content/TestDataset"
DRIVE_ROOT   = "/content/drive/MyDrive"
DRIVE_BACKUP = "/content/drive/MyDrive/WACV_2027_Polyp_Datasets"
# Per-run, via $WACV_RUN: every run has merged into one Drive folder because the backup is a
# copytree(dirs_exist_ok=True), so a new run's last.ckpt overwrites the previous run's in place
# and the restore step then resumes the next run from whatever landed there last. That is how a
# baseline was destroyed once and how W1's NaN epoch-40 last.ckpt became a resume trap. Setting
# WACV_RUN gives the run its own folder, which also makes the restore a no-op instead of a
# silent cross-run resume.
CKPT_BACKUP  = "/content/drive/MyDrive/WACV_2027_Checkpoints"
if os.environ.get("WACV_RUN"):
    CKPT_BACKUP += "_" + os.environ["WACV_RUN"]
WHEEL_CACHE  = "/content/drive/MyDrive/WACV_2027_Wheel_Cache"
CKPT_BACKUP_EVERY_MIN = 15

# PraNet's public archives: the standard 1,450-pair training split and the five test splits.
PRANET_TRAIN_ID = "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb"
PRANET_TEST_ID  = "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao"
# Optional Drive folder holding the same TestDataset.zip + TrainDataset.zip, for when the
# public files hit Google Drive's download quota.
SHARED_FOLDER_ID = os.environ.get("POLYP_DATA_FOLDER_ID", "")

# Ground-truth PraNet benchmark split sizes
EXPECTED_TEST_SPLITS = {
    "CVC-300":           60,
    "CVC-ClinicDB":      62,
    "CVC-ColonDB":       380,
    "ETIS-LaribPolypDB": 196,
    "Kvasir":            100,
}
EXPECTED_TRAIN_IMAGES = 1450


# ─── helpers ──────────────────────────────────────────────────────────────────

def _header(step, title):
    print(f"\n{'='*60}")
    print(f"[{step}] {title}")
    print("=" * 60)


def _count_images(root, recursive=True):
    p = Path(root)
    pattern = "**/*" if recursive else "*"
    return sum(len(list(p.glob(f"{pattern}{ext}"))) for ext in (".png", ".jpg", ".jpeg", ".bmp"))


def _count_dir_images(img_dir: Path):
    """Count images (png+jpg) directly in one dir — mirrors audit_datasets' per-dir counting."""
    if not img_dir.exists():
        return 0
    return len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))


def _train_images_ok():
    """True only if TrainDataset has exactly EXPECTED_TRAIN_IMAGES images (not a partial extract)."""
    if not os.path.exists(TRAIN_ROOT):
        return False
    img_dir, _ = _resolve_img_mask_dirs(Path(TRAIN_ROOT))
    return _count_dir_images(img_dir) == EXPECTED_TRAIN_IMAGES


def _test_images_ok():
    """True only if every test split has its exact expected image count."""
    if not os.path.exists(TEST_ROOT):
        return False
    for split, expected in EXPECTED_TEST_SPLITS.items():
        img_dir, _ = _resolve_img_mask_dirs(Path(TEST_ROOT) / split)
        if _count_dir_images(img_dir) != expected:
            return False
    return True


def _resolve_img_mask_dirs(base: Path):
    img_dir  = base / "images" if (base / "images").exists() else base / "image"
    mask_dir = base / "masks"  if (base / "masks").exists()  else base / "mask"
    return img_dir, mask_dir


def _flatten_nested(root: str, name: str):
    for candidate in [os.path.join(root, name), os.path.join(root, "__MACOSX")]:
        if os.path.isdir(candidate) and candidate != root:
            if "__MACOSX" in candidate:
                shutil.rmtree(candidate)
            else:
                print(f"  Flattening: {candidate}/ → {root}/")
                for item in os.listdir(candidate):
                    src = os.path.join(candidate, item)
                    dst = os.path.join(root, item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                shutil.rmtree(candidate, ignore_errors=True)


# ─── step 1: environment ──────────────────────────────────────────────────────

def check_env():
    _header("1/10", "Environment Check")
    print(f"Python : {sys.version.split()[0]}")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}  |  GPU: {gpu}")
    except ImportError:
        print("PyTorch not yet installed")


# ─── step 2: drive check ──────────────────────────────────────────────────────

def check_drive() -> bool:
    _header("2/10", "Google Drive Check")
    if os.path.isdir(DRIVE_ROOT):
        print(f"Drive mounted at /content/drive — OK")
        return True

    print("Drive NOT mounted.")
    print()
    print("  Run this in a separate Colab cell FIRST, then re-run this script:")
    print()
    print("    from google.colab import drive")
    print("    drive.mount('/content/drive')")
    print()
    print("  Continuing WITHOUT Drive — no backup/restore this session.")
    return False


# ─── step 3: restore checkpoints from drive ───────────────────────────────────

def restore_checkpoints(drive_ok: bool):
    _header("3/10", "Restore Checkpoints From Drive")
    ckpt_dir = os.path.join(REPO_PATH, "lightning_logs")
    if os.path.exists(ckpt_dir):
        ckpts = list(Path(ckpt_dir).glob("**/checkpoints/*.ckpt"))
        print(f"Checkpoints already present ({len(ckpts)} files) — skipping restore.")
    elif not drive_ok:
        print("Drive not mounted — skipping restore.")
        return
    else:
        backup = os.path.join(CKPT_BACKUP, "lightning_logs")
        if not os.path.exists(backup):
            print(f"No checkpoint backup at {CKPT_BACKUP} — starting fresh.")
        else:
            print(f"Restoring lightning_logs from {CKPT_BACKUP} ...")
            os.makedirs(REPO_PATH, exist_ok=True)
            shutil.copytree(backup, ckpt_dir)
            ckpts = list(Path(ckpt_dir).glob("**/checkpoints/*.ckpt"))
            print(f"Restored {len(ckpts)} checkpoint files.")

    if not drive_ok:
        return

    # _save_checkpoints backs runs/ up but nothing ever restored it, so max1b2's resume came up
    # with no history file: the monitor's replay found nothing, the first monitored epoch
    # reopened the JSON in "w" mode and dropped epochs 1-31, and the next 15-min backup pushed
    # that truncation over the Drive copy. The replay in ZeroShotMonitorCallback.__init__ only
    # protects the run if the file is actually on disk. Newer-only, because a fresh clone's
    # committed histories carry a checkout mtime and must not be rolled back to a Drive copy.
    runs_backup, runs_dir = os.path.join(CKPT_BACKUP, "runs"), os.path.join(REPO_PATH, "runs")
    if not os.path.isdir(runs_backup):
        return
    os.makedirs(runs_dir, exist_ok=True)
    restored = []
    for fn in sorted(os.listdir(runs_backup)):
        s, d = os.path.join(runs_backup, fn), os.path.join(runs_dir, fn)
        if not os.path.isfile(s):
            continue
        if os.path.exists(d) and os.stat(d).st_mtime >= os.stat(s).st_mtime:
            continue
        shutil.copy2(s, d)
        restored.append(fn)
    print(f"Restored zero-shot history: {', '.join(restored) if restored else 'nothing newer'}.")


# ─── step 4: clean workspace ──────────────────────────────────────────────────

def clean_workspace():
    _header("4/10", "Clean Workspace")

    cleaned = False
    for name, root in [("TrainDataset", TRAIN_ROOT), ("TestDataset", TEST_ROOT)]:
        if not os.path.exists(root):
            continue
        count = _count_images(root)
        if count == 0:
            print(f"  Removing empty {name} at {root}")
            shutil.rmtree(root)
            cleaned = True
        else:
            print(f"  {name}: {count} total files — keeping.")

    # Remove any stale staging folder from previous partial downloads
    for candidate in Path("/content").iterdir():
        if candidate.is_dir() and candidate.name not in (
            "polp_wacv", "TrainDataset", "TestDataset", "drive", "sample_data"
        ) and not candidate.name.startswith("."):
            # Only remove if it looks like a stale download staging dir
            zips = list(candidate.glob("*.zip"))
            if zips:
                print(f"  Removing stale staging dir with zips: {candidate}")
                shutil.rmtree(str(candidate))
                cleaned = True

    if not cleaned:
        print("  Workspace clean — nothing to remove.")


# ─── step 5: repo setup ───────────────────────────────────────────────────────

def setup_repo():
    _header("5/10", "Repository Setup")
    # A stale non-git REPO_PATH (e.g. a bare lightning_logs dir) would send us down the
    # "pull" branch and fail; wipe it so we clone fresh.
    if os.path.exists(REPO_PATH) and not os.path.exists(os.path.join(REPO_PATH, ".git")):
        print(f"{REPO_PATH} exists but is not a git repo — removing for a clean clone.")
        shutil.rmtree(REPO_PATH)
    if not os.path.exists(REPO_PATH):
        print(f"Cloning {REPO_URL} ({REPO_BRANCH}) ...")
        # userdata.get() only works inside the notebook kernel's own process — this
        # script runs as a `!python ...` subprocess, which has no secrets channel, so
        # userdata.get() silently fails there. Prefer the GITHUB_TOKEN env var (set it
        # in the notebook cell BEFORE invoking this script); fall back to userdata.get()
        # for the case where this is imported/run in-process instead.
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            try:
                from google.colab import userdata
                token = userdata.get("GITHUB_TOKEN")
            except Exception:
                token = None
        url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL
        result = subprocess.run(["git", "clone", "-b", REPO_BRANCH, url, REPO_PATH], capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"Clone failed: {result.stderr}")
        print("Cloned successfully.")
    else:
        # GitHub's default branch (main) can drift arbitrarily far behind the branch this
        # project actually develops on. A bare `git pull` here trusts whatever branch the
        # working tree already happens to be on — correct only because some earlier manual
        # step checked out REPO_BRANCH in this same persistent runtime. A fresh Colab VM has
        # no such history, so pin explicitly every time instead of assuming it.
        print(f"Repo exists — fetching and checking out {REPO_BRANCH} ...")
        fetch = subprocess.run(["git", "-C", REPO_PATH, "fetch", "origin", REPO_BRANCH], capture_output=True, text=True)
        if fetch.returncode != 0:
            sys.exit(f"Fetch failed: {fetch.stderr}")
        checkout = subprocess.run(["git", "-C", REPO_PATH, "checkout", REPO_BRANCH], capture_output=True, text=True)
        if checkout.returncode != 0:
            sys.exit(f"Checkout failed: {checkout.stderr}")
        result = subprocess.run(["git", "-C", REPO_PATH, "pull", "origin", REPO_BRANCH], capture_output=True, text=True)
        print(result.stdout.strip() or "Already up to date.")

    os.chdir(REPO_PATH)
    src_path = os.path.join(REPO_PATH, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    print(f"Working dir: {os.getcwd()} (branch: {REPO_BRANCH})")


# ─── step 6: dependencies ─────────────────────────────────────────────────────

def _mamba_env_fingerprint() -> str:
    """Key the Drive wheel cache by the exact torch/CUDA/Python combo — these are compiled
    CUDA extensions, so a wheel built for one combo is an ABI mismatch for any other. A
    runtime with a different torch/Python version naturally gets its own cache folder
    instead of reusing (and silently misusing) an incompatible one."""
    import torch
    py = f"py{sys.version_info.major}.{sys.version_info.minor}"
    return f"{py}_torch{torch.__version__}".replace("/", "_")


def _install_mamba_stack(drive_ok: bool, no_wheel_cache: bool = False):
    """Install causal-conv1d + mamba-ssm, the real CUDA extensions (never a stub/fallback —
    model.net's Linear+GELU stub only activates if this import fails at train time, which is
    a hard error there, not something this function should ever paper over).

    First run for a given torch/Python combo: builds from source and caches the resulting
    wheels to Drive. Every subsequent run with the same combo: installs straight from the
    cached .whl files (seconds, no compile). `no_wheel_cache=True` forces a from-source build
    even if a cached wheel exists (useful to rule out a stale/corrupt cached wheel as the
    cause of an install failure); the freshly built wheel is still cached afterward."""
    cache_dir = os.path.join(WHEEL_CACHE, _mamba_env_fingerprint())
    cached = (
        [] if no_wheel_cache
        else (list(Path(cache_dir).glob("*.whl")) if drive_ok and os.path.isdir(cache_dir) else [])
    )

    if cached:
        print(f"Found {len(cached)} cached wheel(s) for this environment — installing from Drive (no build):")
        for w in cached:
            print(f"    {w.name}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + [str(w) for w in cached],
            check=True
        )
    else:
        # --no-build-isolation means these builds run in the CURRENT environment rather than
        # an isolated one, so pip does NOT auto-install their build-time deps — ninja and
        # packaging must already be present or the build silently falls back to a slow
        # serial (non-ninja) compile, or crashes outright on `import packaging` in setup.py.
        print("Installing build-time deps (ninja, packaging) ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "ninja", "packaging"],
            check=True
        )

        # mamba-ssm/causal-conv1d's setup.py reads MAX_JOBS to parallelize nvcc compilation of
        # their CUDA kernels; left unset it serializes, which is the main reason this step can
        # look "stuck" for 30-60+ min. Cap at 8 — Colab's compile step OOMs the VM if every
        # core compiles a kernel at once on runtimes with limited RAM per core.
        build_env = os.environ.copy()
        build_env["MAX_JOBS"] = str(min(os.cpu_count() or 2, 8))
        print(f"No cached wheel for this exact torch/Python combo — building from source (MAX_JOBS={build_env['MAX_JOBS']})...")
        # -v (not -q) here on purpose: with -q, pip shows only "Preparing metadata... done" and
        # then goes SILENT for the entire compile (ninja's per-file "[n/N] Building CUDA object"
        # lines are swallowed), which is indistinguishable from a genuine hang. -v streams that
        # output live so a slow-but-progressing build is visibly different from a stalled one.
        local_dir = "/content/_mamba_wheel_build"
        os.makedirs(local_dir, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "-v", "--no-build-isolation", "-w", local_dir,
             "causal-conv1d==1.6.2.post1", "mamba-ssm==2.3.2.post1"],
            check=True, env=build_env
        )
        # Only cache the two packages we actually built, not transitive deps that happen to
        # get wheeled alongside them (e.g. an already-satisfied torch would waste hundreds of
        # MB of Drive space for no benefit — it's not the slow part anyway).
        own_wheels = [w for w in Path(local_dir).glob("*.whl")
                      if w.name.lower().startswith(("causal_conv1d", "mamba_ssm"))]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "--no-index", "--find-links", local_dir, "causal-conv1d", "mamba-ssm"],
            check=True
        )
        if drive_ok:
            os.makedirs(cache_dir, exist_ok=True)
            for w in own_wheels:
                shutil.copy(w, cache_dir)
            print(f"Cached {len(own_wheels)} built wheel(s) to Drive for future sessions: {cache_dir}")
        else:
            print("Drive not mounted — built wheels won't be cached; next fresh runtime will rebuild.")

    # A stale/mismatched cached wheel must fail loudly here, not silently fall through to
    # model.net's CPU stub once training starts.
    check = subprocess.run(
        [sys.executable, "-c", "import causal_conv1d, mamba_ssm"],
        capture_output=True, text=True
    )
    if check.returncode != 0:
        print("FATAL: causal_conv1d/mamba_ssm failed to import after install.")
        print(check.stderr)
        if cached:
            print(f"\nThe cached wheels at {cache_dir} may be stale or built for a different")
            print("environment. Delete that Drive folder to force a fresh rebuild, then re-run.")
        sys.exit(1)


def install_dependencies(drive_ok: bool = False, no_wheel_cache: bool = False):
    _header("6/10", "Dependencies")
    # Always run this: pip no-ops on already-satisfied packages, so this is cheap even on a
    # live runtime — but skipping it entirely (as before) meant a NEW package added to
    # requirements.txt after mamba_ssm was already importable would never get installed
    # (this is exactly how the py_sod_metrics ModuleNotFoundError at epoch 20 happened).
    print("Installing requirements.txt ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

    if importlib.util.find_spec("mamba_ssm") is not None and not no_wheel_cache:
        print("mamba_ssm already importable — skipping causal-conv1d/mamba-ssm install.")
    else:
        _install_mamba_stack(drive_ok, no_wheel_cache)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
    print("Dependencies installed.")


# ─── step 7: download + extract datasets ──────────────────────────────────────

def _extract_zip(zip_path: str, dest_dir: str, name: str):
    print(f"  Extracting {os.path.basename(zip_path)} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    _flatten_nested(os.path.join(dest_dir, name), name)
    _flatten_nested(os.path.join(dest_dir, name), "__MACOSX")


def download_datasets():
    _header("7/10", "Dataset Download")
    train_ok = _train_images_ok()
    test_ok  = _test_images_ok()

    if train_ok and test_ok:
        print("  Both datasets already present (exact counts) — skipping.")
        return

    # Try to restore from Drive backup first (fast path, avoids re-download)
    if os.path.isdir(DRIVE_BACKUP):
        for name, dst in [("TrainDataset", TRAIN_ROOT), ("TestDataset", TEST_ROOT)]:
            src = os.path.join(DRIVE_BACKUP, name)
            if os.path.exists(src) and not os.path.exists(dst):
                count = _count_images(src)
                if count > 0:
                    print(f"  Restoring {name} from Drive backup ({count} images)...")
                    shutil.copytree(src, dst)
        train_ok = _train_images_ok()
        test_ok  = _test_images_ok()
        if train_ok and test_ok:
            print("  Both datasets restored from Drive backup (exact counts).")
            return

    staging = Path("/content/_polyp_staging")
    staging.mkdir(exist_ok=True)

    if SHARED_FOLDER_ID:
        url = f"https://drive.google.com/drive/folders/{SHARED_FOLDER_ID}"
        print(f"  Downloading shared folder ({SHARED_FOLDER_ID})...")
        result = subprocess.run(["gdown", "--folder", url, "-O", str(staging)])

        # gdown --folder -O <dir> places the Drive folder INSIDE <dir>
        # Find the downloaded subfolder (should be one new dir inside staging)
        dl_dirs = [p for p in staging.iterdir() if p.is_dir()]
        dl_root = dl_dirs[0] if dl_dirs else staging
    else:
        print("  Downloading PraNet's public TrainDataset.zip and TestDataset.zip ...")
        for name, file_id in [("TrainDataset", PRANET_TRAIN_ID), ("TestDataset", PRANET_TEST_ID)]:
            subprocess.run(["gdown", file_id, "-O", str(staging / f"{name}.zip")])
        dl_root = staging

    print(f"  Staging contents: {[p.name for p in dl_root.iterdir()]}")

    for zip_file in sorted(dl_root.glob("*.zip")):
        name = zip_file.stem  # "TestDataset" or "TrainDataset"
        target_root = Path(f"/content/{name}")
        already_ok = _train_images_ok() if name == "TrainDataset" else _test_images_ok() if name == "TestDataset" else _count_images(str(target_root)) > 0
        if target_root.exists() and already_ok:
            print(f"  {name} already extracted (exact counts) — skipping.")
            continue
        mb = zip_file.stat().st_size // (1024 * 1024)
        print(f"  Extracting {zip_file.name} ({mb} MB)...")
        try:
            with zipfile.ZipFile(str(zip_file), "r") as z:
                z.extractall("/content/")
        except Exception as e:
            print(f"  ERROR extracting {zip_file.name}: {e}")
            continue
        _flatten_nested(str(target_root), name)
        count = _count_images(str(target_root))
        print(f"  {name}: {count} images ready.")

    # Clean up staging
    shutil.rmtree(str(staging), ignore_errors=True)

    # Final counts
    tc = _count_images(TRAIN_ROOT) if os.path.exists(TRAIN_ROOT) else 0
    sc = _count_images(TEST_ROOT)  if os.path.exists(TEST_ROOT)  else 0
    print(f"\n  TrainDataset: {tc} total files  |  TestDataset: {sc} total files")


# ─── step 8: dataset structure audit ─────────────────────────────────────────

def audit_datasets() -> bool:
    _header("8/10", "Dataset Structure Audit")
    all_ok = True

    # ── TestDataset ───────────────────────────────────────────
    print(f"\n  [TestDataset]  {TEST_ROOT}")
    if not os.path.exists(TEST_ROOT):
        print("    MISSING")
        return False

    found_splits = sorted(
        d for d in os.listdir(TEST_ROOT)
        if os.path.isdir(os.path.join(TEST_ROOT, d)) and not d.startswith("__")
    )
    missing = [s for s in EXPECTED_TEST_SPLITS if s not in found_splits]
    extra   = [s for s in found_splits if s not in EXPECTED_TEST_SPLITS]
    if missing:
        print(f"    MISSING splits: {missing}")
        all_ok = False
    if extra:
        print(f"    Extra/unexpected splits: {extra}")

    print(f"\n    {'Split':<25} {'images':>7} {'masks':>7} {'expect':>7} {'paired':>7}  status")
    print(f"    {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  ------")

    for split in sorted(EXPECTED_TEST_SPLITS):
        expected = EXPECTED_TEST_SPLITS[split]
        if split not in found_splits:
            print(f"    {split:<25} {'---':>7} {'---':>7} {expected:>7} {'---':>7}  MISSING")
            all_ok = False
            continue

        split_path = Path(TEST_ROOT) / split
        img_dir, mask_dir = _resolve_img_mask_dirs(split_path)

        if not img_dir.exists() or not mask_dir.exists():
            print(f"    {split:<25}  missing {'images/' if not img_dir.exists() else 'masks/'} dir")
            all_ok = False
            continue

        imgs  = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        masks = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
        img_stems  = [p.stem for p in imgs]
        mask_stems = [p.stem for p in masks]
        paired = img_stems == mask_stems
        count_ok = len(imgs) == expected
        status = "OK" if paired and count_ok else "FAIL"

        print(f"    {split:<25} {len(imgs):>7} {len(masks):>7} {expected:>7} {str(paired):>7}  {status}")

        if not count_ok:
            print(f"      FAIL: expected {expected} images, got {len(imgs)} (partial/corrupt extract)")
            all_ok = False
        if not paired:
            img_set, mask_set = set(img_stems), set(mask_stems)
            orphan_imgs  = img_set - mask_set
            orphan_masks = mask_set - img_set
            if orphan_imgs:
                print(f"      images without mask : {sorted(orphan_imgs)[:5]}")
            if orphan_masks:
                print(f"      masks without image : {sorted(orphan_masks)[:5]}")
            all_ok = False

    # ── TrainDataset ──────────────────────────────────────────
    print(f"\n  [TrainDataset]  {TRAIN_ROOT}")
    if not os.path.exists(TRAIN_ROOT):
        print("    MISSING")
        return False

    train_path = Path(TRAIN_ROOT)
    img_dir, mask_dir = _resolve_img_mask_dirs(train_path)

    if not img_dir.exists():
        print(f"    MISSING image dir (tried image/ and images/)")
        all_ok = False
    elif not mask_dir.exists():
        print(f"    MISSING mask dir (tried masks/ and mask/)")
        all_ok = False
    else:
        imgs  = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        masks = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
        img_stems  = [p.stem for p in imgs]
        mask_stems = [p.stem for p in masks]
        paired  = img_stems == mask_stems

        print(f"    Dirs      : {img_dir.name}/ ({len(imgs)} files) + {mask_dir.name}/ ({len(masks)} files)")
        print(f"    Paired    : {paired}")
        print(f"    Split     : full set, no val holdout ({len(imgs)} train, PraNet convention)")
        print(f"    Extensions: images={set(p.suffix for p in imgs)}  masks={set(p.suffix for p in masks)}")

        if len(imgs) != EXPECTED_TRAIN_IMAGES:
            print(f"    FAIL: expected {EXPECTED_TRAIN_IMAGES} images, got {len(imgs)} (partial/corrupt extract)")
            all_ok = False
        if not paired:
            img_set, mask_set = set(img_stems), set(mask_stems)
            orphans = img_set.symmetric_difference(mask_set)
            print(f"    FAIL: {len(orphans)} unpaired files")
            all_ok = False

    # ── Summary ───────────────────────────────────────────────
    print()
    if all_ok:
        print("  All dataset checks PASSED.")
    else:
        print("  Some checks FAILED.")
        ans = input("  Continue training anyway? [y/N]: ").strip().lower()
        if ans != "y":
            sys.exit("Aborted after audit failures.")

    return all_ok


# ─── step 9: save datasets to drive ──────────────────────────────────────────

def _copy_tree_incremental(src, dst):
    """copytree(dirs_exist_ok=True) rewrites every file on every pass. The checkpoint backup
    runs every 15 min over a directory holding three ~420 MB files, only one of which changes
    between the 20-epoch marks, so an unfiltered copy pushes ~1.3 GB into the Drive FUSE mount
    each time — ~18 GB over a 3.5 h run, most of it re-copying bytes that did not change. Drive
    keeps the superseded versions and they count against the quota. Skipping unchanged files
    cuts that by roughly 3x. Matching on (size, mtime) is safe here because copy2 preserves
    mtime, which is also the key train.py resumes on."""
    copied = skipped = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for fn in files:
            s, d = os.path.join(root, fn), os.path.join(out, fn)
            st = os.stat(s)
            if os.path.exists(d):
                dt = os.stat(d)
                if dt.st_size == st.st_size and int(dt.st_mtime) == int(st.st_mtime):
                    skipped += 1
                    continue
            shutil.copy2(s, d)
            copied += 1
    return copied, skipped


def save_datasets_to_drive(drive_ok: bool):
    _header("9/10", "Save Datasets to Drive")
    if not drive_ok:
        print("Drive not mounted — skipping save.")
        return

    os.makedirs(DRIVE_BACKUP, exist_ok=True)
    saved = False
    for name, src in [("TrainDataset", TRAIN_ROOT), ("TestDataset", TEST_ROOT)]:
        dst = os.path.join(DRIVE_BACKUP, name)
        if os.path.exists(dst):
            print(f"  {name} already in Drive backup — skipping.")
            continue
        if os.path.exists(src) and _count_images(src) > 0:
            print(f"  Saving {name} to Drive ({_count_images(src)} images)...")
            shutil.copytree(src, dst)
            saved = True

    if not saved:
        print("  All datasets already backed up to Drive.")

    _lock = threading.Lock()

    def _save_checkpoints(quiet=False):
        with _lock:
            os.makedirs(CKPT_BACKUP, exist_ok=True)
            # `runs/` holds the zero-shot history JSON, which audit_accuracy.py --history needs
            # to report the best-epoch-on-test selection rule. Backing up lightning_logs alone
            # loses it on every runtime recycle, and it cannot be regenerated from the ckpt.
            totals = []
            for name in ("lightning_logs", "runs"):
                src = os.path.join(REPO_PATH, name)
                if not os.path.exists(src):
                    continue
                dst = os.path.join(CKPT_BACKUP, name)
                if not quiet:
                    print(f"\nSaving {name} to Drive: {dst}")
                copied, skipped = _copy_tree_incremental(src, dst)
                totals.append(f"{name}: {copied} copied, {skipped} unchanged")
            if not quiet:
                print("Checkpoint backup complete (" + "; ".join(totals) + ").")

    # atexit alone is not enough: it runs on a clean exit or a KeyboardInterrupt, but a Colab
    # runtime that is recycled or hard-disconnected kills the process outright and the handler
    # never fires. That is how the acceptance-gate run's checkpoints were lost. A periodic
    # daemon copy bounds the loss to CKPT_BACKUP_EVERY_MIN of training instead of all of it.
    def _periodic():
        while True:
            time.sleep(CKPT_BACKUP_EVERY_MIN * 60)
            try:
                _save_checkpoints(quiet=True)
            except Exception as exc:
                print(f"[backup] periodic checkpoint copy failed: {type(exc).__name__}: {exc}")

    atexit.register(_save_checkpoints)
    threading.Thread(target=_periodic, daemon=True).start()
    print(f"  Checkpoint backup registered -> {CKPT_BACKUP} (on exit/interrupt, and every "
          f"{CKPT_BACKUP_EVERY_MIN} min).")
    if not os.environ.get("WACV_RUN"):
        print("  [warn] WACV_RUN is unset, so this run shares the default checkpoint folder "
              "with every other run and will overwrite its last.ckpt. Set it per run.")


# ─── step 10: training ────────────────────────────────────────────────────────

def _build_train_cmd(args, extra_args):
    # The launcher owns exactly one flag (--fresh, which skips the Drive restore and tells
    # train.py to wipe local lightning_logs/ — it does NOT clear CKPT_BACKUP, which is
    # merged into with dirs_exist_ok=True, so archive that folder by hand before reusing it)
    # and forwards everything else verbatim. Mirroring train.py's flags here is what caused
    # the drift this replaces: the launcher kept passing --gate-engage-*/--boundary-loss-weight
    # after train.py dropped them, which is an argparse crash at launch. train.py's argparser
    # is now the single source of truth.
    # -u is not cosmetic: subprocess.run() below inherits the notebook cell's stdout, which is a
    # pipe rather than a TTY, so CPython block-buffers it at 8 KB. One monitored epoch prints
    # ~2 KB, so output arrives in bursts of ~4 epochs — silence for ~25 min at 6.4 min/epoch,
    # and a burst that cuts mid-eval wherever the 8 KB boundary lands. That reads as a hang.
    cmd = [sys.executable, "-u", "src/train.py"]
    if args.fresh:
        cmd.append("--fresh")
        print("--fresh: will wipe lightning_logs/ and train from scratch.")
    else:
        print("Auto-resumes from latest checkpoint if one exists.")
    if extra_args:
        cmd += extra_args
        print(f"[launcher] Forwarding to train.py: {' '.join(extra_args)}")
    return cmd


def launch_training(args, extra_args):
    _header("10/10", "Launch Training")
    cmd = _build_train_cmd(args, extra_args)
    print(f"\n[launcher] Resolved training command: {' '.join(cmd)}\n")
    if args.dry_run:
        print("[launcher] --dry-run set, exiting without launching training.")
        return
    # Don't restate train.py's defaults here — this line said "every 20 epochs" while the
    # forwarded --zeroshot-every 1 made it fire every epoch. train.py owns its own flags.
    result = subprocess.run(cmd, cwd=REPO_PATH)
    sys.exit(result.returncode)


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument("--fresh", action="store_true", help="Delete checkpoints and train from scratch")
    _parser.add_argument("--no-wheel-cache", action="store_true",
                          help="Force a from-source rebuild of causal-conv1d/mamba-ssm, ignoring any cached Drive wheel")
    # Deliberately no training flags here — see _build_train_cmd. Everything else
    # (--model, --recipe, --seed, --epochs, --aug, ...) is forwarded to train.py untouched.
    _parser.add_argument("--dry-run", action="store_true",
                          help="Print the resolved train.py command and exit without launching training")
    _args, _extra = _parser.parse_known_args()
    # parse_known_args leaves a `--` separator in the leftovers rather than consuming it, and
    # train.py's parser then rejects the whole tail as unrecognized. The runbook documents the
    # `--` form, so strip it here instead of making the caller remember not to type it.
    while _extra and _extra[0] == "--":
        _extra = _extra[1:]

    check_env()
    drive_ok = check_drive()
    clean_workspace()
    setup_repo()  # must clone REPO_PATH before restore_checkpoints writes lightning_logs into it
    if not _args.fresh:
        restore_checkpoints(drive_ok)
    install_dependencies(drive_ok, no_wheel_cache=_args.no_wheel_cache)
    download_datasets()
    audit_datasets()
    save_datasets_to_drive(drive_ok)
    launch_training(_args, _extra)
