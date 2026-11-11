"""Uniform params/MACs measurement for every vendored competitor, one thop call each.

Published cost columns disagree across papers by up to 2x (Polyp-PVT 10.1 G in Agnihotri
WACV'26 vs 5.30 G in BAMPolyp Tbl 4; PraNet 13.1 vs 6.93), so no published column can be
mixed with ours. This re-measures every model we hold source for at one resolution, one
batch size, one thop call, and writes the JSON that docs/analysis/efficiency_measured.md
is built from.

Params and MACs are weight-independent, so pretrained loading is patched out — three of the
four vendored constructors hard-code a checkpoint path from the original authors' machines.
"""

import argparse
import contextlib
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]

# (vendor root, module, factory, note). Channel widths are the ones each paper's own
# __main__/train script constructs, not our defaults.
MODELS = {
    "polyp_pvt": (
        REPO / "Polyp-PVT-main" / "Polyp-PVT-main",
        "lib.pvt",
        lambda m: m.PolypPVT(),
        "channel=32; the control for the whole cost column (published 10.1 G)",
    ),
    "hgm": (
        REPO / "HGM-main",
        "lib.pvt",
        lambda m: m.HGM(channel=32, device="cuda" if torch.cuda.is_available() else "cpu"),
        "channel=32; needs mamba_ssm + torch_geometric; only model above us with code and no published cost",
    ),
    "fccnet": (
        REPO / "FCC-Net-main" / "FCC-Net-main",
        "lib.FCCNet",
        lambda m: m.FCCNet(channel=32, pretrained=False),
        "channel=32; the origin of our octave-conv decoder",
    ),
    "pranet": (
        REPO / "PraNet-master" / "PraNet-master",
        "lib.PraNet_Res2Net",
        lambda m: m.PraNet(channel=32),
        "Res2Net-50 backbone; anchors against three mutually inconsistent published values",
    ),
    "ctnet": (
        REPO / "CTNet-main" / "CTNet-main",
        "model.Mymodel13",
        lambda m: m.PolypModule(channel=32),
        "MiT-b3 backbone; highest-scoring model in our table that ships code AND prediction maps",
    ),
    "cfanet": (
        REPO / "CFANet-main" / "CFANet-main",
        "lib.model",
        lambda m: m.CFANet(channel=64),
        "Res2Net-50 two-stream + boundary branch; channel=64 is what its own train.py builds",
    ),
    "pranet_v2": (
        REPO / "PraNet-V2-main" / "PraNet-V2-main" / "multiclass_seg" / "binary_seg",
        "lib.pranet",
        lambda m: m.PVT_PraNet_V2(channel=32, num_class=1, use_softmax=False),
        "PVT-PraNet-V2, the PVTv2-b2 variant — same encoder as ours, so the closest cost peer",
    ),
    "emcad": (
        REPO / "PraNet-V2-main" / "PraNet-V2-main" / "multiclass_seg" / "EMCAD",
        "lib.networks",
        lambda m: m.EMCADNet(num_classes=1, pretrain=False),
        "EMCADNet, the backbone Polyp-DiFoM's best row is built on; vendored inside PraNet-V2",
    ),
    "ours": (None, None, None, "SpectraMambaNet at train.py's defaults"),
}


@contextlib.contextmanager
def _no_pretrained():
    """Params and MACs do not depend on weights, and three constructors hard-code a
    checkpoint path from the original authors' filesystem (`/media/nercms/...`)."""
    real_load, real_lsd = torch.load, nn.Module.load_state_dict
    torch.load = lambda *a, **k: {}
    nn.Module.load_state_dict = lambda self, *a, **k: None
    try:
        yield
    finally:
        torch.load, nn.Module.load_state_dict = real_load, real_lsd


@contextlib.contextmanager
def _timm_registry_frozen():
    """Undo vendored repos' hijacking of timm's global model registry.

    Polyp-PVT's and FCC-Net's `lib/pvtv2.py` both `@register_model` their own `pvt_v2_b0..b5`,
    which *overwrites* timm's entrypoints for the whole process. `_vendor_path`'s sys.modules
    cleanup does not undo that, so profiling `ours` after either of them made
    `timm.create_model('pvt_v2_b2', features_only=True)` return Polyp-PVT's class, which ignores
    `features_only` — an AttributeError here, but on another timm version it would silently
    profile a different encoder and quietly corrupt the paper's cost column.
    """
    try:
        from timm.models import _registry as reg
    except ImportError:  # timm <0.9 kept these in timm.models.registry
        from timm.models import registry as reg
    # deepcopy, not copy: `_module_to_models` is a dict of *sets*, and a shallow copy would
    # share those sets with the live registry. deepcopy treats functions and classes as atomic,
    # so the entrypoint values are not duplicated.
    saved = {k: copy.deepcopy(v) for k, v in vars(reg).items()
             if k.startswith(("_model", "_module")) and isinstance(v, (dict, set, list))}
    try:
        yield
    finally:
        # Mutate in place — other timm modules hold direct references to these containers.
        for k, v in saved.items():
            cur = getattr(reg, k)
            cur.clear()
            if isinstance(cur, list):
                cur.extend(v)
            else:
                cur.update(v)


@contextlib.contextmanager
def _vendor_path(root):
    """Each vendored repo ships its own top-level `lib` package, so importing a second one
    in the same process would silently reuse the first one's modules.

    `src` also has to come off the path for the duration. CTNet's `model/` and `utils/` have
    no `__init__.py`, so they are namespace portions; our `src/model/__init__.py` makes ours a
    *regular* package, and a regular package anywhere on the path beats a namespace portion
    regardless of order. Leaving `src` in place resolves `model.Mymodel13` against our own
    package and raises ModuleNotFoundError.
    """
    shadowed = [p for p in sys.path if p and Path(p).resolve() == (REPO / "src")]
    for p in shadowed:
        sys.path.remove(p)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        sys.path.extend(shadowed)
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if f and Path(f).resolve().is_relative_to(root):
                del sys.modules[name]
                continue
            # Namespace packages (CTNet's `model/` and `utils/` have no __init__.py) have
            # __file__ = None, so the check above misses them and the stale entry then
            # shadows our own `src/model` on the next model in the loop. torch.ops and
            # torch.classes expose a __path__ that is not iterable, hence the try.
            try:
                paths = list(getattr(mod, "__path__", ()) or ())
            except TypeError:
                continue
            if any(Path(p).resolve().is_relative_to(root) for p in paths):
                del sys.modules[name]


def _build_ours(device, **overrides):
    sys.path.insert(0, str(REPO / "src"))
    from model.net import SpectraMambaNet

    # SpectraMambaNet's constructor defaults `use_fcc_decoder=False` while train.py passes
    # True, so building it bare profiles the legacy decoder — a different network from the
    # one in the accuracy table. Mirror train.py's defaults explicitly.
    cfg = dict(pretrained_encoder=False, use_fcc_decoder=True, fcc_channel=64,
               fcc_region_calib=True, edge_stride=8, mamba_stage="bottleneck",
               use_mamba_bottleneck=True, encoder_name="pvt_v2_b2")
    cfg.update(overrides)
    return SpectraMambaNet(**cfg).to(device)


def _coerce_kv(kv):
    k, _, v = kv.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"--ours-flags expects K=V, got {kv!r}")
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"
    try:
        return k, int(v)
    except ValueError:
        return k, v


def profile_one(name, size, device, repeat, ours_overrides=None):
    from thop import profile as thop_profile

    root, module, factory, note = MODELS[name]
    row = {"model": name, "note": note, "size": size, "device": str(device)}

    if name == "ours":
        model = _build_ours(device, **(ours_overrides or {}))
        row["overrides"] = ours_overrides or {}
    else:
        import importlib

        with _timm_registry_frozen(), _vendor_path(root), _no_pretrained():
            model = factory(importlib.import_module(module)).to(device)

    model.eval()
    row["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6

    dummy = torch.randn(1, 3, size, size, device=device)
    try:
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        row["gmacs"] = macs / 1e9
        row["gflops_2x"] = 2 * macs / 1e9
    except Exception as e:  # thop cannot hook custom CUDA kernels or GCN scatter ops
        row["gmacs"] = None
        row["thop_error"] = f"{type(e).__name__}: {e}"

    # Both conventions, every model, so the choice of published column is made on numbers
    # rather than on assumption. They are NOT interchangeable: thop hooks nn.Module.forward and
    # therefore misses every raw matmul, which on a transformer means the attention itself.
    # Measured here at 352 on CPU: PraNet (pure conv) agrees to -0.7 %, but Polyp-PVT comes out
    # +7.4 % and CTNet +8.3 % under dispatch. The published figures track THOP -- CTNet's own
    # 15.14 reproduces as 15.139 -- so a table that prints our dispatch number beside their
    # transcribed one is comparing two different measurements. Publish one column or the other,
    # measured the same way for every row.
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(dummy)
        row["gmacs_dispatch"] = counter.get_total_flops() / 2 / 1e9
    except Exception as e:
        row["gmacs_dispatch"] = None
        row["dispatch_error"] = f"{type(e).__name__}: {e}"

    if repeat:
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
        row["fps"] = repeat / (time.time() - start)

    del model
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--repeat", type=int, default=0,
                        help="Forward passes for FPS. Default 0 — CPU FPS is not comparable "
                             "to any published GPU figure, so only pass this on the GPU.")
    parser.add_argument("--out", type=str, default=None, help="Write results as JSON")
    parser.add_argument("--ours-flags", nargs="+", default=[], metavar="K=V",
                        help="SpectraMambaNet constructor overrides for the 'ours' row, e.g. "
                             "mamba_scan=raster mamba_bidirectional=true fcc_region_proto=true. "
                             "Needed whenever the trained config differs from train.py's "
                             "defaults, because those flags change the parameter count.")
    args = parser.parse_args()

    ours_overrides = dict(_coerce_kv(kv) for kv in args.ours_flags)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in args.models:
        try:
            row = profile_one(name, args.size, device, args.repeat, ours_overrides)
        except Exception as e:
            row = {"model": name, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        print(json.dumps(row))

    print(f"\n{'model':<12} {'Params M':>9} {'GMACs':>9} {'GFLOPs':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<12} {'FAILED':>9}  {r['error'][:60]}")
            continue
        g = r.get("gmacs")
        print(f"{r['model']:<12} {r['params_m']:>9.2f} "
              f"{(f'{g:.2f}' if g else 'n/a'):>9} {(f'{2 * g:.2f}' if g else 'n/a'):>9}")

    ppvt = next((r for r in rows if r["model"] == "polyp_pvt" and r.get("gmacs")), None)
    if ppvt:
        # The gate: our GMACs may only be quoted next to Agnihotri's column if their
        # "GFLOPs" really is the thop MAC count. 10.1 G is their Polyp-PVT figure.
        dev = abs(ppvt["gmacs"] - 10.1) / 10.1
        print(f"\n[gate] Polyp-PVT {ppvt['gmacs']:.2f} GMACs vs published 10.1 — "
              f"{dev:+.1%} — {'PASS' if dev <= 0.05 else 'FAIL: rebuild the cost table'}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
