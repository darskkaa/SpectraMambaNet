"""Re-derive every claim in MANIFEST.md from the files in this repository.

Nothing is read from MANIFEST.md itself. Exits non-zero on the first mismatch.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

PASS, FAIL = [], []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    (PASS if ok else FAIL).append(f"{name}: got {got!r} want {want!r}")
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ('' if ok else f"  (expected {want!r})"))


# The fin2 [run-config], as the run printed it.
KW = dict(
    encoder_name='pvt_v2_b2', use_fcc_decoder=True, fcc_region_calib=True,
    fcc_share_bn=False, fcc_channel=64, fcc_reduce_octave=False,
    fcc_region_proto=True, use_rfp=False, edge_stride=4,
    mamba_stage='h4h8', mamba_scan='raster', mamba_freq_axis=True,
    mamba_ls_init=1e-4, use_mamba_bottleneck=True, ds_mutation=True,
    mamba_bidirectional=True, img_size=352, fullres_head=False,
    pretrained_encoder=False,
)
CKPT = os.path.join(ROOT, 'fin2_soup8.ckpt')
CKPT_SHA = '598de21496dbc7f7c0619ea0d2401c7dc28b6d5885c638359d695e0c69401e0b'

# The ablation networks (configs/ablation_commands.sh) as constructor arguments, each with the
# deployed parameter count the paper prints. R3 is R2's network trained on different data.
_BLOCK = dict(mamba_stage='h4h8', mamba_scan='raster', mamba_bidirectional=True)
_GAIN = dict(freq_gain='tanh', freq_gain_scale=1.0, freq_gain_shared=True)
ARMS = {
    'R0': (dict(use_fcc_decoder=False, fcc_region_calib=False, use_mamba_bottleneck=False,
                edge_stride=8), 31_469_175),
    'R1': (dict(use_fcc_decoder=True, fcc_region_calib=False, edge_stride=8, **_BLOCK),
           34_766_085),
    'R2': (dict(use_fcc_decoder=True, fcc_region_proto=True, edge_stride=4, **_BLOCK),
           34_854_150),
    'M0': (dict(use_fcc_decoder=True, fcc_channel=80, fcc_region_proto=True, edge_stride=4,
                use_mamba_bottleneck=False), 34_930_646),
    'M5': (dict(use_fcc_decoder=True, fcc_region_proto=True, edge_stride=4, **_BLOCK, **_GAIN),
           34_833_672),
    'M3': (dict(use_fcc_decoder=True, fcc_region_proto=True, edge_stride=4, third_axis='raster',
                **_BLOCK, **_GAIN), 34_833_672),
}
# What logs/ablation_R0-R3.txt must print for each row of the paper's ablation table: the five
# Dice cells (Kvasir, ClinicDB, CVC-300, ColonDB, ETIS), the five-set mean, GMACs, and the
# training-time parameter count, which carries the contrast head when --contrast-weight > 0.
ABLATION = {
    'R0': ((0.9053, 0.9244, 0.8926, 0.8053, 0.8056), 0.8666, 11.57, False),
    'R1': ((0.9150, 0.9139, 0.8896, 0.7974, 0.7996), 0.8631, 15.68, False),
    'R2': ((0.9159, 0.9325, 0.8872, 0.7988, 0.7896), 0.8648, 16.37, True),
    'R3': ((0.9196, 0.9226, 0.8909, 0.8126, 0.8188), 0.8729, 16.37, True),
}
FIVE = ('Kvasir', 'CVC-ClinicDB', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB')


def section(t):
    print(f"\n=== {t} ===")


def verify_config():
    section('A. config — the command and the printed run-config agree')
    cfg = json.load(open(os.path.join(ROOT, 'configs', 'fin2_run_config.json')))
    check('epochs', cfg['epochs'], '160')
    check('encoder', cfg['encoder'], 'pvt_v2_b2')
    check('ema_decay (no EMA)', cfg['ema_decay'], '0.0')
    check('val_fraction (no holdout)', cfg['val_fraction'], '0.0')
    check('select_on', cfg['select_on'], 'indomain')
    check('select_weights (raw, not EMA)', cfg['select_weights'], 'raw')
    check('lr_schedule', cfg['lr_schedule'], 'cosine-restarts')
    check('restart_cycle_len', cfg['restart_cycle_len'], '20')
    check('mamba_stage', cfg['mamba_stage'], 'h4h8')
    check('mamba_scan', cfg['mamba_scan'], 'raster')


def verify_model(with_ckpt):
    section('B. model — parameter counts and the 35.088 / 34.854 reconciliation')
    from model.net import SpectraMambaNet
    train_model = SpectraMambaNet(**KW, use_contrast_head=True)
    infer_model = SpectraMambaNet(**KW, use_contrast_head=False)
    n_train = sum(p.numel() for p in train_model.parameters())
    n_infer = sum(p.numel() for p in infer_model.parameters())

    check('trained params (log :464 prints 35.088 M)', n_train, 35_088_262)
    check('deployed params (the cost table)', n_infer, 34_854_150)
    check('difference is the contrast head', n_train - n_infer, 234_112)

    by_child = {k: sum(p.numel() for p in v.parameters())
                for k, v in train_model.named_children()}
    check('encoder', by_child['encoder'], 24_849_856)
    check('mamba_h8', by_child['mamba_h8'], 749_568)
    check('mamba_h4', by_child['mamba_h4'], 208_896)
    check('fcc_decoder', by_child['fcc_decoder'], 9_045_830)
    check('contrast_head', by_child['contrast_head'], 234_112)
    check('children sum to total', sum(by_child.values()), n_train)

    if not with_ckpt:
        print('  -- skipping checkpoint checks (--no-ckpt)')
        return

    section('C. weights — hash, metadata, and a strict load against this code')
    import torch
    h = hashlib.sha256()
    with open(CKPT, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    check('sha256(fin2_soup8.ckpt)', h.hexdigest(), CKPT_SHA)
    check('size in bytes', os.path.getsize(CKPT), 140_669_680)

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    check('epoch (0-indexed)', ck['epoch'], 79)
    check('global_step', ck['global_step'], 21600)
    check('state_dict tensors', len(ck['state_dict']), 863)
    members = [re.search(r'epoch=(\d+)', m).group(1) for m in ck['soup_members']]
    check('soup members (0-indexed epochs)', members, ['079', '119'])

    sd = {(k[6:] if k.startswith('model.') else k): v for k, v in ck['state_dict'].items()}
    try:
        train_model.load_state_dict(sd, strict=True)
        strict_ok = True
    except RuntimeError:
        strict_ok = False
    check('loads strict=True against the trained construction', strict_ok, True)

    try:
        infer_model.load_state_dict(sd, strict=True)
        infer_strict = True
    except RuntimeError:
        infer_strict = False
    check('rejects strict=True without the head (expected)', infer_strict, False)
    missing, unexpected = infer_model.load_state_dict(sd, strict=False)
    check('only the head is unexpected', all(k.startswith('contrast_head.') for k in unexpected), True)
    check('nothing is missing', list(missing), [])


def verify_selection():
    section('D. selection — the eight cycle ends, and the disproof of test selection')
    evals = json.load(open(os.path.join(ROOT, 'results', 'fin2_eval_trajectory.json')))
    check('evaluation points', len(evals), 32)

    cyc = [e for e in evals if e['epoch'] % 20 == 0]
    check('cycle-end snapshots', len(cyc), 8)
    expect = {20: 0.9234, 40: 0.9224, 60: 0.9211, 80: 0.9242,
              100: 0.9158, 120: 0.9228, 140: 0.9132, 160: 0.9196}
    for e in cyc:
        check(f'  in-domain @ep{e["epoch"]}', round(e['indomain'], 4), expect[e['epoch']], 1e-9)

    pool_best = max(cyc, key=lambda e: e['indomain'])
    check('pool argmax on in-domain', pool_best['epoch'], 80)
    check('its five-set mean', round(pool_best['five'], 4), 0.8753, 1e-9)

    gbest = max(evals, key=lambda e: e['indomain'])
    check('global in-domain argmax is NOT ep80', gbest['epoch'], 150)
    check('  its five-set mean is lower', round(gbest['five'], 4), 0.8642, 1e-9)
    check('  so the rule did not read the test sets', gbest['five'] < pool_best['five'], True)


def verify_results():
    section('E. results — the five headline rows and the maps behind them')
    d = json.load(open(os.path.join(ROOT, 'results', 'fin2_soup8_scored.json')))
    check('scored model', d['model'], 'fin2_soup8')
    # every cell printed in README.md's results table
    want = {
        'Kvasir':            (0.9274, 0.8838, 0.9193, 36.56, 100),
        'CVC-ClinicDB':      (0.9386, 0.8935, 0.9584, 6.81, 62),
        'CVC-300':           (0.8980, 0.8326, 0.9589, 13.37, 60),
        'CVC-ColonDB':       (0.8308, 0.7531, 0.8670, 39.30, 380),
        'ETIS-LaribPolypDB': (0.8219, 0.7446, 0.9202, 102.95, 196),
    }
    total = 0.0
    for name, (dice, iou, rec, hd, n) in want.items():
        s = d['splits'][name]
        total += s['mDice']
        check(f'  {name} mDice', round(s['mDice'], 4), dice, 1e-9)
        check(f'  {name} mIoU', round(s['mIoU'], 4), iou, 1e-9)
        check(f'  {name} Recall', round(s['Recall'], 4), rec, 1e-9)
        check(f'  {name} HD95', round(s['HD95'], 2), hd, 1e-9)
        check(f'  {name} images', s['n_images'], n)
        check(f'  {name} no missing preds', s['n_missing_preds'], 0)
        check(f'  {name} per-image entries', len(s['per_image']), n)
    check('five-set mean', round(total / 5, 6), 0.883335, 1e-9)

    section('F. prediction maps on disk match the scored counts')
    for name, spec in want.items():
        n = spec[-1]
        p = os.path.join(ROOT, 'predictions', name)
        check(f'  {name}/', len(os.listdir(p)), n)


def verify_ablation():
    section('G. ablation — the R0-R3 log, and every arm built from this code')
    from model.net import SpectraMambaNet
    built = {}
    for arm, (kw, want) in ARMS.items():
        m = SpectraMambaNet(encoder_name='pvt_v2_b2', pretrained_encoder=False, **kw)
        built[arm] = sum(p.numel() for p in m.parameters())
        check(f'{arm} builds at', built[arm], want)
    built['R3'] = built['R2']
    check('M3 and M5 are parameter-matched', built['M3'], built['M5'])
    check('M0, block removed for decoder width, is larger than R3', built['M0'] > built['R3'], True)

    log = open(os.path.join(ROOT, 'logs', 'ablation_R0-R3.txt'), encoding='utf-8').read()
    parts = re.split(r'^# ARM (R\d) ', log, flags=re.M)
    body = dict(zip(parts[1::2], parts[2::2]))
    check('arms in the log', sorted(body), sorted(ABLATION))
    means = {}
    for arm, (cells, mean, gmacs, head) in ABLATION.items():
        b = body[arm]
        done = re.search(rf'ARM {arm} COMPLETE(.*?)={{20}}', b, re.S).group(1)
        dice = dict(re.findall(r'^\s+(\S+)\s+mDice ([\d.]+)$', done, re.M))
        check(f'  {arm} Dice cells', tuple(float(dice[s]) for s in FIVE), cells)
        means[arm] = float(re.findall(r'mean over all 5 splits[^:]*: ([\d.]+)', b)[-1])
        check(f'  {arm} five-set mean at epoch 40', means[arm], mean, 1e-9)
        check(f'  {arm} GMACs', round(float(re.search(r'^GMACs:\s+([\d.]+)', b, re.M).group(1)), 2),
              gmacs, 1e-9)
        check(f'  {arm} params as profiled', float(re.search(r'params\s*:\s*([\d.]+) M', done).group(1)),
              round(built[arm] / 1e6, 2), 1e-9)
        n_train = built[arm] + (234_112 if head else 0)
        check(f'  {arm} params at train time', float(re.search(r'\[params\] total=([\d.]+) M', b).group(1)),
              round(n_train / 1e6, 3), 1e-9)

    for (a, b), want in {('R0', 'R1'): -0.0035, ('R1', 'R2'): 0.0017, ('R2', 'R3'): 0.0081,
                         ('R0', 'R3'): 0.0063}.items():
        check(f'step {a} -> {b}', round(means[b] - means[a], 4), want, 1e-9)

    evals = json.load(open(os.path.join(ROOT, 'results', 'fin2_eval_trajectory.json')))
    final = max(evals, key=lambda e: e['epoch'])
    check('fin2 final-epoch five-set mean (the averaging paragraph)', round(final['five'], 4),
          0.8701, 1e-9)


def verify_significance():
    section('H. significance — the paired tests, re-run from the per-image scores')
    shipped = json.load(open(os.path.join(ROOT, 'results', 'significance_fin2.json')))['splits']
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'sig.json')
        r = subprocess.run([sys.executable, os.path.join('src', 'paired_significance.py'),
                            '--ours', os.path.join('results', 'fin2_soup8_scored.json'),
                            '--out', out], cwd=ROOT, capture_output=True,
                           env={**os.environ, 'PYTHONUTF8': '1'})
        check('src/paired_significance.py exits 0', r.returncode, 0)
        if r.returncode:
            return
        rerun = json.load(open(out))['splits']

    keys = sorted((s, m) for s in shipped for m in shipped[s])
    check('re-run covers the same splits and competitors',
          keys, sorted((s, m) for s in rerun for m in rerun[s]))
    fields = ('n', 'wins', 'mean', 'median', 'p', 'p_holm')
    bad = [(s, m, k) for s, m in keys for k in fields
           if not math.isclose(rerun[s][m][k], shipped[s][m][k], rel_tol=1e-6, abs_tol=1e-12)]
    check(f'all {len(keys) * len(fields)} values match results/significance_fin2.json', bad, [])

    def sig_wins(split):
        return {m for m, x in shipped[split].items() if x['p_holm'] < 0.05 and x['mean'] > 0}
    check('competitors with released maps', len(shipped['Kvasir']), 11)
    check('significant on all five datasets against', len(set.intersection(*map(sig_wins, FIVE))), 8)
    check('  against all eleven on Kvasir', len(sig_wins('Kvasir')), 11)
    check('  against all eleven on CVC-ColonDB', len(sig_wins('CVC-ColonDB')), 11)
    check('datasets where a competitor leads on mean',
          sorted({s for s in FIVE for x in shipped[s].values() if x['mean'] < 0}), ['CVC-300'])
    check('  none of those leads is significant',
          [m for m, x in shipped['CVC-300'].items() if x['mean'] < 0 and x['p_holm'] < 0.05], [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-ckpt', action='store_true',
                    help='skip everything that needs fin2_soup8.ckpt')
    a = ap.parse_args()
    with_ckpt = not a.no_ckpt and os.path.exists(CKPT)
    if not a.no_ckpt and not with_ckpt:
        print(f"note: {CKPT} not found — running with --no-ckpt semantics.\n"
              f"      Download it from the release assets to run the full check.\n")

    verify_config()
    verify_model(with_ckpt)
    verify_selection()
    verify_results()
    verify_ablation()
    verify_significance()

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} FAILED")
    for f in FAIL:
        print(f"  FAIL {f}")
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
