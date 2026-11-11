# PROVENANCE — every number in the paper, and the file it came from

Written 2026-08-13 against that day's draft of the paper, and edited on 2026-11-02 for this
repository. The edit annotates references to working files that are not shipped, removes three
sections about the draft's own editing, and lists below what the paper changed after the
draft. Nothing was re-measured for it. Line numbers refer to the draft's LaTeX sources, which
are not in this repository, so they locate a passage rather than a line of the published PDF.

**Purpose.** A question about any printed figure should be answerable from a file, not from
memory. Every number in the draft is listed below with the file that produced it.
[NUMBER_AUDIT.md](NUMBER_AUDIT.md) is the check of this ledger: it re-derived 334 printed values
from the raw files without trusting this one, re-read every `tab:main` row against its source
paper, and found no value mismatch. §6.1 below gives the intended source of each row and
`NUMBER_AUDIT.md` §3 what re-reading it returned.

### Changed after this ledger was written

The accepted and camera-ready versions changed the printed values below. Every other number in
this file appears in the final paper as written here.

| where | 2026-08-13 draft | final paper | why |
|---|---|---|---|
| `tab:main`, IC-PolypSeg | .910 .938 .909 .807 .774 | .910 .938 .907 .799 .758 | the draft row came from a secondary table (HAT-SAM3, s00371 p7) that printed a per-dataset maximum over eight EfficientNet variants (EF7/EF7/EF2/EF6/EF4). The final row is IC-PolypSeg's single-model EF7 row from its own arXiv 2309.08234, Tables 1-2 (2026-08-27) |
| `tab:main`, CVC-300 marks | IC-PolypSeg .909 bold, CTNet .908 underlined | CTNet .908 bold, IC-PolypSeg .907 underlined | follows from the row above |
| `tab:main`, new row | none | Polyp-DiFoM, EMCAD-backbone configuration: .909 .958 .889 .776 .759 | Agnihotri et al., WACV 2026, Tables 1-2 p6. Its .958 takes the CVC-ClinicDB bold, EMCAD's .944 becomes the underline, and ours (.939) is third. The table now has 17 competitor rows |
| `tab:cost`, CTNet | 44.29 / 15.14, measured on our harness | 44.19‡ / 15.20‡, CTNet's own Table II | our measurement stays in `results/efficiency.json` (44.290657 M / 15.139 G) |
| `tab:cost`, new row | none | Polyp-DiFoM 26.76 / 10.58 | it ships in the EMCAD architecture, so it repeats the EMCAD measurement |
| `tab:cost`, our row | 34.85 / 16.37 (18.77) | 34.85 / 16.37 | the analytic scan floor left the table and its caption on 2026-08-15; the caption now says the traced 16.37 is what the convention yields for a state-space model. 18.773 stays in `results/efficiency.json` as `gmacs_with_scan_floor`, and the notes in that file and in `results/COST.md` that the paper prints `16.37 (18.77)` predate the change |
| results prose, CVC-300 | below the best by 0.011 (.909 − .8980) | below the best encoder-scale entry by 0.010 (.908 − .8980) | follows from the IC-PolypSeg correction |
| results prose, new | none | "in-domain we lead on Kvasir and are third on CVC-ClinicDB"; on the per-image test "we lead eight of the eleven methods it covers and tie the remaining three" | the only negative mean differences in `tab:sig` (Polyp-PVT −.0022 and CTNet −.0102, both CVC-300) are not significant (§6.3) |
| results prose, SAM-Mamba | "over twenty times our compute" | "an order of magnitude more compute" | SAM-Mamba's 423 may be GFLOPs, which would put the MAC ratio near 12.9×, so the prose now claims only the order of magnitude (2026-08-27) |
| results prose, measure ranges | S_α, E_φ, F^w_β and MAE ranges (§6.2) | removed | the supplement prints all six measures per dataset |
| abstract | the five Dice values, the PraNet deltas, "more than twelve Dice points", 16.37 GMACs | none of these. It keeps 34.85 M and 26.4 FPS and adds "roughly half the tokens of a full-spectrum scan" and "against eleven re-scored methods the model leads eight and ties the rest" | rewritten before submission and edited again for the camera-ready version |
| introduction, "more than twelve points" | best ClinicDB .944 (EMCAD) against best ETIS .822 (ours): 12.2 | best ClinicDB is now .958 (Polyp-DiFoM), 14.5 and 14.8 points above the best published CVC-ColonDB and ETIS entries (CTNet .813, .810), and 12.7 and 13.6 above ours (.831, .822) | the sentence holds either way |
| setup, estimator check | "within 0.0005" | "within 0.001" | the measured maximum error is unchanged at 0.000500 |
| ablations, new | none | "R1 and R2 together net −0.0018" (.8648 − .8666), and the spectral arm's sequence is "1.95× shorter" | arithmetic on §7.1 and §4 |
| conclusion | .8980 against a best .909 on CVC-300, the five Dice values, the eight-of-eleven count | removed; CVC-300 is now "where the lead does not hold". Added: the 26.4 FPS is on a datacentre L4, within 25-30 FPS endoscopy streams, and suite hardware is unmeasured; every score is from an in-domain-selected checkpoint; R2 adds the head together with its losses and deep supervision, so the head's own contribution is not isolated. The single seed-0 statement and the 0.0040 spread are unchanged | |
| supplement, new | none | layer shapes read from `src/model/net.py` and `src/model/fcc_modules.py`, and the failure gallery's count of frames below 0.1 per-image Dice: 1 of 100 on Kvasir, 8 of 380 on CVC-ColonDB, 10 of 196 on ETIS-Larib, none on CVC-ClinicDB or CVC-300 | the counts are the `per_image` entries of `results/fin2_soup8_scored.json` |

### Files cited below, and where they are in this repository

| cited as | shipped as |
|---|---|
| `run_final_2.txt` | `logs/run_final_2.txt`, line numbers unchanged |
| `ABLATION_sweep.txt` | `logs/ablation_R0-R3.txt`, line numbers unchanged |
| `runs/fin2_soup8_scored.json` | `results/fin2_soup8_scored.json` |
| `runs/significance_fin2.json` | `results/significance_fin2.json` |
| `docs/analysis/predmaps/<method>.json` | `results/competitors/<method>.json` |
| `docs/analysis/polyppvt_predmaps.json`, `ctnet_predmaps.json` | `results/competitors/polyppvt.json`, `ctnet.json` |
| `docs/analysis/efficiency_measured.json` | `results/efficiency.json`: the same parameter and MAC values for every competitor it keeps, with the stale `ours` record (§6.5) replaced by the shipped model's; `results/COST.md` explains why |

Every other file in this table is byte-identical to the development repository's copy
(checked 2026-11-02), so each line number cited for it holds here. Everything else cited below, the rest of
`docs/analysis/`, `docs/paper/` and `docs/runs/`, `ablation_run_1.txt` and the source PDFs in
`papers/`, is a working file of the private development repository and is not shipped. Where a
claim rests on one of those, `scripts/verify_release.py` re-derives what it can from the shipped
files (`MANIFEST.md` §9), and the cited papers are named by table and page so they can be read
at the source.

---

## 0. The four source classes

| class | meaning | files |
|---|---|---|
| **RUN** | printed by a training/scoring run on GPU | `run_final_2.txt`, `ABLATION_sweep.txt`, `ablation_run_1.txt`. The first two ship as `logs/run_final_2.txt` and `logs/ablation_R0-R3.txt`, with line numbers unchanged |
| **SCORED** | produced by our estimator over prediction maps | `runs/*.json`, `docs/analysis/*predmaps.json` |
| **MEASURED** | counted on CPU by a profiler, no training | `docs/analysis/code_audit.md` (not shipped; `scripts/verify_release.py` §B and §G rebuild the parameter counts it records for the shipped model and every ablation arm), `docs/analysis/efficiency_measured.json` |
| **TRANSCRIBED** | copied from another paper's own table | `docs/analysis/literature_scores.md` (not shipped), each row carrying its source paper and page, which §6.1 repeats |

Nothing in the paper is taken from a planning note. Where a note and a run log disagreed, the
run log won.

---

## 1. The shipped model, and what "shipped" means

The checkpoint behind every "Ours" row is `fin2_soup8`: a greedy weight-space average of 2 of the
8 cosine-restart snapshots of the `fin2` run, selected on the in-domain splits only.

| fact | value | source |
|---|---|---|
| full flag set | `epochs=160 bs=16 lr=1e-4 wd=1e-4 aug=endo hat_prob=0.5 lr_schedule=cosine-restarts warmup=5 restart_cycle_len=20 multi_scale=True use_fcc_decoder=True fcc_channel=64 img_size=352 mamba_stage=h4h8 mamba_freq_axis=True mamba_bidirectional=True mamba_ls_init=1e-4 ema_decay=0.0 boundary_dice=0.3 contrast_weight=0.1 precision=bf16-mixed` | `run_final_2.txt:459` `[run-config]` |
| fresh start, not resumed | `[setup] FRESH start — no existing backup` | `run_final_2.txt:254` |
| GPU | NVIDIA L4, PyTorch 2.11.0+cu128 | `run_final_2.txt:265` |
| wall clock | `exit 0 after 8.42 h` | `run_final_2.txt:1115` |
| in-domain selection signal | mean of the Kvasir + CVC-ClinicDB **test** splits, e.g. `selected on raw indomain 0.8806` | `run_final_2.txt:510` and each later `Saved ... best-on-test.ckpt` line |
| the explicit warning that this is the published convention | `--val-fraction 0 - no held-out split ... pass --select-on-test to reproduce the published max-over-epochs-on-test rule` | `run_final_2.txt:465` |

**Our five headline Dice values, and all six threshold-swept measures, come from one file:**
`runs/fin2_soup8_scored.json`.

| split | mDice | mIoU | S_α | E_φ | F^w_β | MAE | Recall@.5 | HD95@.5 | n |
|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG | .9274 | .8838 | .9363 | .9697 | .9257 | .0183 | .9193 | 36.56 | 100 |
| CVC-ClinicDB | .9386 | .8935 | .9531 | .9826 | .9367 | .0059 | .9584 | 6.81 | 62 |
| CVC-300 | .8980 | .8326 | .9345 | .9657 | .8794 | .0069 | .9589 | 13.37 | 60 |
| CVC-ColonDB | .8308 | .7531 | .8827 | .9181 | .8164 | .0266 | .8670 | 39.30 | 380 |
| ETIS-Larib | .8219 | .7446 | .8909 | .9208 | .7810 | .0155 | .9202 | 102.95 | 196 |

Every one of these appears in the paper, in `tab:main` (rounded to three places), `tab:hd95`
(the Ours row), `sec/5_results.tex:49-56`, `sec/8_conclusion.tex:27`, and both supplement tables.

**Cost.** The shipped configuration is arm **R3** of the cumulative sweep, confirmed by its
`[config]` line at `ABLATION_sweep.txt:1423`.

| fact | value | source |
|---|---|---|
| parameters | 34.85 M | `ABLATION_sweep.txt:1424` (also `:1060` for R2, same config) |
| parameters, six decimals | 34.854150 M | `docs/analysis/code_audit.md:57`; rebuilt from `src/` by `scripts/verify_release.py` §B |
| traced GMACs, `thop`, 352², bs 1 | 16.366 | `ABLATION_sweep.txt:1425` |
| untraced selective-scan analytic floor | 2.407 G | `results/efficiency.json`: `gmacs_with_scan_floor` 18.773 − `gmacs_traced` 16.366 |
| traced count plus the floor | 16.366 + 2.407 = 18.773 | `results/efficiency.json`. The draft's `tab:cost` printed it as (18.77); since 2026-08-15 the paper prints the traced 16.37 alone |
| throughput | 26.4 FPS, bs 1, 352², 100 reps, L4 | `ABLATION_sweep.txt:1427` and `:1063` |

---

## 2. Abstract — `sec/0_abstract.tex`

These are the draft abstract's numbers. The camera-ready abstract keeps only 34.85 M, 26.4 FPS
and the counting claims; see the header.

| line | printed | source |
|---|---|---|
| 3-4 | "more than twelve Dice points" | `tab:main`: best encoder-scale CVC-ClinicDB is 0.944 (EMCAD), best encoder-scale ETIS is 0.822 (ours). 12.2 points. |
| 6 | 34.85 M, 16.37 GMAC, 26.4 FPS | §1 cost block |
| 9 | "halving that sequence at no representational loss" | Hermitian symmetry of the rFFT of a real map: `H·(W/2+1)` bins against `H·W` positions. Mechanism, not a measurement. |
| 12-13 | .927 / .939 / .898 / .831 / .822 | `runs/fin2_soup8_scored.json` |
| 14 | +.029 / +.040 / +.027 / +.119 / +.194 | `runs/significance_fin2.json`, `PraNet-released` mean paired per-image delta, per split |
| 15-17 | "significant on all five against eight of eleven, and against all eleven on Kvasir and CVC-ColonDB" | `runs/significance_fin2.json`, counted in §6.3 below |
| 17-18 | "a parameter-matched spatial control" | §7.2 below |

---

## 3. Introduction — `sec/1_intro.tex`

| line | printed | source |
|---|---|---|
| 35 | "trails ... by more than twelve points" | same as abstract line 3-4. With Polyp-DiFoM's .958 in the final table the gap is wider; see the header |
| 41-43 | "a 2026 audit of 27 papers ... significance test in one of them, boundary-distance or sensitivity metrics in two apiece" | `docs/analysis/psrc_scorecard.md`: C1 25/27 omit HD95, C2 25/27 omit Recall, C5 26/27 report no significance test |
| 47 | 34.85 M, 16.37 GMACs | §1 |
| 64 | 0.36 M for the spectral axis | `docs/analysis/code_audit.md:378-388` — `--no-freq-axis` gives 34.493318 M, delta −0.360832 |
| 64 | 1.95× shorter | 9 680 / 4 972 = 1.947 |
| 64 | 16 % fewer scan tokens | 1 − 48 664 / 58 080 = 16.2 % |
| 68 | 34.8337 M matched arms | `docs/analysis/code_audit.md:378-388` — `--third-axis raster` delta is 0.000000 |
| 68 | 0.0013 mean Dice between arms | §7.2 |
| 74 | "four of the five items" | `docs/analysis/psrc_scorecard.md`, "Score: 4 of 5. No audited paper exceeds 2." |
| 75 | +11.9 on CVC-ColonDB, +19.4 on ETIS | `runs/significance_fin2.json`, PraNet row: +.1192 and +.1935 |

---

## 4. Method — `sec/3_method.tex`

Architecture facts are read off `src/model/net.py`; the two counted quantities are measured.

| line | printed | source |
|---|---|---|
| 9-10 | strides 4/8/16/32, channels 64/128/320/512, grids 88²/44²/22²/11² | PVTv2-B2 definition; grids follow from a 352² input |
| 16-17 | 10.00 M saved by placing the blocks on `f_1`/`f_0` rather than the 11² bottleneck; 34.85 M against 44.86 M | `docs/analysis/code_audit.md:57` — 44.856582 against 34.854150, difference 10.002432 |
| 31 | LayerScale initialised at 1e-4 | `run_final_2.txt:459`, `mamba_ls_init=0.0001` |
| 36 | scans run outside AMP | `src/model/net.py`; method choice following SegMamba |
| 49 | clip bound s = 3 | `ABLATION_sweep.txt:333` `[config] freq_gain_scale=3.0` |
| 64 | 4 972 against 9 680 tokens at the two stages | half-spectra 44·(44/2+1) + 88·(88/2+1) = 1 012 + 3 960 = 4 972 against spatial grids 44² + 88² = 9 680; the per-block count is `last_scan_tokens` at `src/model/net.py:246-249` |
| 64 | 1.95× | 9 680 / 4 972 |
| 65-66 | 48 664 against 58 080 whole-network scan tokens, a 16 % reduction | `net.py:246-249`; the arithmetic is in `NUMBER_AUDIT.md` §2.3 |
| 67 | block adds 0.96 M, 2.7 % of the network | `docs/analysis/code_audit.md:432` |
| 67 | spectral axis is 0.36 M of that | `docs/analysis/code_audit.md:378-388` |
| 80 | 34.8337 M, parameter-identical to six decimals | `docs/analysis/code_audit.md:378-388` |
| 81 | control scans a sequence 1.95× longer | same ratio, inverted |
| 106-107 | sharpening temperature τ = 0.25 | `src/model/net.py` calibration head |
| 130 | a radius fixed at stride 8 spans roughly 48 input pixels | 6 feature pixels × stride 8 |
| 130-131 | "would swallow whole any polyp under ~50 px across, which is most of ETIS-Larib and CVC-ColonDB" | design rationale; the size distribution is not printed as a number |
| 155 | λ_e = 0.3, λ_c = 0.1 | `run_final_2.txt:459`, `boundary_dice=0.3 contrast_weight=0.1` |
| 164 | "Two of eight snapshots survive" | `runs/fin2_soup8_scored.json` model id, and the greedy loop's own log |

---

## 5. Experimental setup — `sec/4_experiments.tex`

| line | printed | source |
|---|---|---|
| 8-13 | 1450 train (900 Kvasir + 550 ClinicDB); test 100 / 62 / 60 / 380 / 196 | PraNet protocol; the five test counts are re-confirmed by `n_images` in `runs/fin2_soup8_scored.json` and by the dataset audit block at `ABLATION_sweep.txt:94-98` |
| 18-23 | 352², AdamW at 1e-4 / 1e-4, batch 16, 160 epochs, bf16-mixed, 5 warm-up epochs, cosine restarts of period 20, scales 0.75/1/1.25 | `run_final_2.txt:459` |
| 30 | about 8.5 h on one L4 GPU, single seed | `run_final_2.txt:1115` (`8.42 h`) and `:265` (L4) |
| 42-43 | harness returns 10.02 GMACs for Polyp-PVT against a published 10.1, and 13.15 for PraNet against 13.1 | `docs/analysis/efficiency_measured.json`: `polyp_pvt` 10.018439 G / 25.107604 M; `pranet` 13.149645 G / 32.547319 M |
| 47 | eleven methods' maps re-scored | the eleven rows of `tab:sig`; nine in `docs/analysis/predmaps/*.json` plus `ctnet_predmaps.json` and `polyppvt_predmaps.json` |
| 49-50 | recovered a published column to within 0.0005 on all five (the final paper says "within 0.001") | `docs/analysis/ctnet_predmaps.json` against CTNet's published row: max absolute error 0.000500. The wider validation table is `docs/analysis/released_maps_table.md` (Polyp-PVT 0.0004, SANet 0.0004, PraNet 0.0026, U-Net++ 0.0015, U-Net 0.0082) |
| 53-55 | per-image scalar is the mean of that image's 256-threshold Dice curve; the identity is asserted to 1e-9 | `src/score_pred_maps.py` assertion |
| 58-59 | scoring one method's maps against themselves returns exactly 0 at p = 1.0 on every split | `docs/analysis/significance_ctnet_selftest.json`: median 0.0, mean 0.0, p 1.0 on all six split keys |
| 60 | a published pair returns p = 3.6e-3 on ETIS-Larib and ties elsewhere | `docs/analysis/significance_ctnet_selftest.json`, Polyp-PVT vs CTNet: ETIS raw 1.789e-3, Holm 3.578e-3; CVC-300 Holm .1265; ColonDB Holm .8438; Kvasir Holm 1; ClinicDB Holm 1 |

---

## 6. Results — `sec/5_results.tex`

### 6.1 `tab:main`, the mean-Dice comparison

Every competitor row is **TRANSCRIBED** from `docs/analysis/literature_scores.md:62-82`, a
working table of the development repository (not shipped) that names the paper and page each row
was read from; the last column below repeats it, and `NUMBER_AUDIT.md` §3.1 expands the short
names. All sixteen rows of the draft reconcile digit for digit. The IC-PolypSeg row below is the
final paper's, and the Polyp-DiFoM row was added after the draft:

| row | Kva / Cli / 300 / Col / ETIS | source (paper, page) |
|---|---|---|
| U-Net | .818 .823 .710 .512 .398 | Dutta p6-7 |
| U-Net++ | .821 .794 .707 .483 .401 | Dutta p6-7 |
| PraNet | .898 .899 .871 .709 .628 | Dutta p6-7 |
| SANet | .904 .916 .888 .753 .750 | Dutta p6-7 |
| MSNet | .907 .921 .869 .755 .719 | Dutta p6-7 |
| HSNet | .926 .937 .903 .810 .808 | s00371 p7 |
| M²SNet | .912 .922 .903 .758 .749 | Dutta p6-7 |
| PVT-Cascade | .911 .919 .892 .781 .786 | Dutta p6-7 |
| CFA-Net | .915 .933 .893 .743 .732 | PR 140:109555 Tbl 1-3 (primary) |
| Polyp-PVT | .917 .937 .900 .808 .787 | Dutta p6-7; Agnihotri p6; 2604.17865 p4-5 |
| MEGANet | .913 .938 .899 .793 .739 | Dutta p6-7 |
| EMCAD | .894 .944 .878 .760 .744 | Agnihotri p6 |
| IC-PolypSeg | .910 .938 .907 .799 .758 | own arXiv 2309.08234 T1-2, EF7 row (s00371 p7 printed a cross-variant max, .910 .938 .909 .807 .774; corrected 2026-08-27) |
| CTNet | .917 .936 .908 .813 .810 | Dutta p6-7 |
| Polyp-DiFoM | .909 .958 .889 .776 .759 | Agnihotri T1-2 p6, EMCAD-backbone configuration (added after the draft) |
| SAM-Mamba | .924 .942 .920 .853 .848 | Dutta p6-7 |
| HAT-SAM3 | .929 .896 .929 .849 .815 | s00371 p7 |

Bold and underline are column best and second within the encoder-scale group only, as the caption
states. Verified: our row takes the bold on Kvasir (.927 over HSNet .926), CVC-ColonDB (.831 over
CTNet .813) and ETIS (.822 over CTNet .810). In the draft it took the underline on CVC-ClinicDB
(.939 under EMCAD .944); in the final table Polyp-DiFoM's .958 is bold there, EMCAD's .944
underlined, and ours third. It takes neither mark on CVC-300, where CTNet .908 is bold and
IC-PolypSeg .907 underlined since the 2026-08-27 EF7 correction; before it, the composite .909
held the bold. No our-row number is bolded that is not the column maximum.

The HAT-SAM3 row is its **out-of-domain** table, not the paper's own "zero-shot" experiment; the
distinction is recorded in the development repository's literature table
(`literature_scores.md:89-93`).

### 6.2 Results prose, `sec/5_results.tex`

| line | printed | source |
|---|---|---|
| 49-51 | .9274 / .9386 / .8980 / .8308 / .8219 | `runs/fin2_soup8_scored.json` |
| 52 | +.029 +.040 +.027 +.119 +.194 against PraNet | `runs/significance_fin2.json`, PraNet row |
| 54-56 | S_α .8827 to .9531, E_φ .9181 to .9826, F^w_β .7810 to .9367, MAE .0059 to .0266 | column ranges of the §1 table. The final paper drops the ranges and points to the supplement, which prints every value |
| 60-61 | +0.018 on CVC-ColonDB, +0.012 on ETIS over the strongest published entry | `tab:main`: .831 − .813 and .822 − .810, both against CTNet |
| 63 | .8980 on CVC-300, below the best by 0.011; the final says "below the best encoder-scale entry by 0.010" | .909 (the composite IC-PolypSeg row) − .8980 in the draft; .908 (CTNet) − .8980 since the 2026-08-27 EF7 correction |
| 64 | "smallest test set at 60 images" | `n_images` in the scored JSON |
| 66-67 | 103 M and 423 GMACs, three times our parameters and over twenty times our compute; the final says "an order of magnitude more compute" | `literature_scores.md` SAM-Mamba row, transcribed; 103.0 / 34.85 = 2.96, and 423 / 18.77 = 22.5 with the floor (423 / 16.37 = 25.8 traced). Reworded 2026-08-27: the 423 may be GFLOPs, which would halve the MAC ratio to about 12.9×, so the prose now claims only the order of magnitude |

### 6.3 `tab:sig`, per-image significance

Every cell is **SCORED**, from `runs/significance_fin2.json`, key `splits[dataset][method]`, field
`mean` for the delta and `p_holm` for the star. Reproduced in full and matching the printed table:

| method | Kvasir | ClinicDB | CVC-300 | ColonDB | ETIS | sig |
|---|---|---|---|---|---|---|
| SFA | +.2044* | +.2382* | +.4306* | +.3750* | +.5246* | 5/5 |
| U-Net | +.1091* | +.1160* | +.1880* | +.3270* | +.4235* | 5/5 |
| U-Net++ | +.1065* | +.1443* | +.1910* | +.3493* | +.4209* | 5/5 |
| DCRNet | +.0411* | +.0424* | +.0416* | +.1268* | +.2663* | 5/5 |
| ACSNet | +.0292* | +.0566* | +.0353* | +.1144* | +.2437* | 5/5 |
| PraNet | +.0292* | +.0396* | +.0270* | +.1192* | +.1935* | 5/5 |
| EU-Net | +.0197* | +.0368* | +.0612* | +.0751* | +.1350* | 5/5 |
| SANet | +.0234* | +.0230* | +.0098* | +.0782* | +.0719* | 5/5 |
| MSEG | +.0309* | +.0292 | +.0238 | +.0958* | +.1223* | 3/5 |
| Polyp-PVT | +.0100* | +.0018 | −.0022 | +.0225* | +.0351 | 2/5 |
| CTNet | +.0103* | +.0031 | −.0102 | +.0181* | +.0121 | 2/5 |

Counting: **eight** methods carry a star in all five columns (SFA, U-Net, U-Net++, DCRNet, ACSNet,
PraNet, EU-Net, SANet); **all eleven** carry one on Kvasir and on CVC-ColonDB. Both claims at
`sec/5_results.tex:105-106` are exactly these counts.

The ETIS sentence at `:109-111` rests on two fields the table does not print: CTNet's ETIS
`mean` is +0.0121 with `median` **−0.0003** and `p_holm` 0.5884 over n = 196. A positive mean with
a negative median is a right-tail effect, which is what that sentence says.

### 6.4 `tab:hd95`, boundary distance and sensitivity

Every competitor cell is **SCORED** by us; none is transcribed, because none of these papers
reports either metric. Sources: `docs/analysis/predmaps/{sfa,unet,unetpp,dcrnet,acsnet,pranet,eunet,mseg,sanet}.json`,
plus `docs/analysis/polyppvt_predmaps.json` and `docs/analysis/ctnet_predmaps.json`. All 110 cells
were re-read against their JSONs and match. The two most-cited rows:

| method | HD95 Kva/Cli/300/Col/ETIS | Recall Kva/Cli/300/Col/ETIS | file |
|---|---|---|---|
| Polyp-PVT | 41.49 / 8.30 / 20.76 / 54.78 / 126.76 | .9133 .9495 .9469 .8205 .8676 | `polyppvt_predmaps.json` |
| CTNet | 41.53 / 7.15 / 12.87 / 46.60 / 127.39 | .9093 .9426 .9447 .8182 .8708 | `ctnet_predmaps.json` |
| Ours | 36.56 / 6.81 / 13.37 / 39.30 / 102.95 | .9193 .9584 .9589 .8670 .9202 | `runs/fin2_soup8_scored.json` |

Prose at `:154-163`: "best on four of five and second on the fifth" holds (CVC-300 13.37 against
CTNet 12.87). "39.30 against 46.60" and "102.95 against 126.76" are the ColonDB and ETIS
comparisons above; 1 − 102.95/126.76 = 18.8 %, printed as 19 %.

### 6.5 `tab:cost`

Unmarked rows are **MEASURED** on our harness, from `results/efficiency.json`: Polyp-PVT
25.107604 M / 10.018439 G, EMCAD 26.764411 M / 10.579868 G, CFA-Net 25.830318 M / 55.360476 G,
PraNet 32.547319 M / 13.149645 G, and in the draft CTNet 44.290657 M / 15.139329 G. The final
table prints CTNet's own Table II figures instead, 44.19‡ / 15.20‡; our measurement stays in the
file. Polyp-DiFoM, added after the draft, ships in the EMCAD architecture and repeats the EMCAD
measurement. Rows carrying † are **TRANSCRIBED**, from Polyp-DiFoM's Table 1 p6 (Agnihotri et
al.), and were collected in `docs/analysis/literature_scores.md:139-144`.

Our row is §1's cost block: 34.85 / 16.37 (18.77) in the draft, 34.85 / 16.37 in the final. The
26.4 FPS in the prose at `:211` is `ABLATION_sweep.txt:1427`, measured on the same L4 as training.

**Two parameter counts that are not this row.** The development repository's efficiency file
also carried an `ours` record at **35.315142 M / 15.679 GMACs**: the model at `train.py`'s
argparse defaults rather than the shipped flags, profiled on CPU where the scan is a stub. It is
not shipped, and `results/COST.md` says why. The shipped numbers are the R2/R3 lines of
`ABLATION_sweep.txt`. **35.088 M**, at `logs/run_final_2.txt:464`, is the training-time model,
which carries a 234,112-parameter contrastive head that inference discards (`MANIFEST.md` §5).

### 6.6 `sec:psrc`

| printed | source |
|---|---|
| boundary distance met in 2 of 27; Recall 2 of 27; per-image significance 1 of 27; no paper meets more than two; we meet four | `docs/analysis/psrc_scorecard.md` (C1 25/27 omit HD95, C2 25/27 omit Recall, C5 26/27 report none, sole exception PolyMamba-Net 2026; C4 external set met by 1/27, TransNetR) |
| "transcription errors in two of four papers spot-checked" and "about a third of the apparent four-year improvement on Kvasir attributable to the split" | the same audit, recorded at `docs/analysis/literature_scores.md:20-35` |

Item 4, the external dataset, is not met and is printed as ✗; the supplement gives the reason.

---

## 7. Ablations — `sec/6_ablations.tex`

### 7.1 `tab:ablation`, the cumulative R0 → R3 sweep

Source: `ABLATION_sweep.txt`, one arm per block, 40 epochs at seed 0, scored at `last.ckpt`.

| arm | per-dataset line | cost lines | Kva / Cli / 300 / Col / ETIS | 5-set mean |
|---|---|---|---|---|
| R0 | `:325-329` | `:334-337` (31.47 M, 11.566 G, 50.0 FPS) | .9053 .9244 .8926 .8053 .8056 | .8666 |
| R1 | `:688-692` | `:697-700` (34.77 M, 15.683 G, 27.5 FPS) | .9150 .9139 .8896 .7974 .7996 | .8631 |
| R2 | `:1051-1055` | `:1060-1063` (34.85 M, 16.366 G, 26.4 FPS) | .9159 .9325 .8872 .7988 .7896 | .8648 |
| R3 | `:1415-1419` | `:1424-1427` (34.85 M, 16.366 G, 26.4 FPS) | .9196 .9226 .8909 .8126 .8188 | .8729 |

Derived quantities in the prose, all arithmetic on the rows above:

- step deltas −0.0035, +0.0017, +0.0081, and endpoint +0.0063 (`:40-42`)
- R1 costs 3.30 M and 4.11 G over R0 (34.77 − 31.47; 15.683 − 11.566), R2 costs 0.08 M and 0.68 G
  over R1 (34.85 − 34.77; 16.366 − 15.683), R3 costs nothing on either axis (`:46-50`). The
  final paper prints R2's step as 0.69 GMACs, the difference of the rounded 16.37 and 15.68
- R1 and R2 together net −0.0018 (.8648 − .8666), a sentence the final paper adds
- R3's `[config]` at `:1423` is what confirms R3 is the shipped configuration

The **0.0040 same-config resolution** quoted in the caption and twice in the prose is the observed
spread between two runs of one configuration at the same SHA, recorded in a planning note of the
development repository (`docs/analysis/ablation_plan_fin2.md` §7, not shipped).
`scripts/verify_release.py` does not re-derive it.

### 7.2 The third-axis control

| printed | value | source |
|---|---|---|
| both arms at 34.8337 M | `--third-axis raster` parameter delta 0.000000 | `docs/analysis/code_audit.md:378-388`; `scripts/verify_release.py` §G checks both arms at 34,833,672, and their commands are in `configs/ablation_commands.sh` |
| spectral arm .9281 .9314 .8890 .8195 .7954, mean .8727 | arm **M5** | the session's output, recorded in a note |
| spatial control .9177 .9223 .8921 .8210 .8170, mean .8740 | arm **M3** | the session's output, recorded in a note |
| difference 0.0013 in favour of the control | .8740 − .8727 | arithmetic |
| "the spectral arm leading on both in-domain splits and the control on all three unseen centres" | .9281 > .9177, .9314 > .9223; .8890 < .8921, .8195 < .8210, .7954 < .8170 | arithmetic on the two rows |

These two arms ran on 2026-08-11 in a separate session, about two hours each on one L4, whose
console output was not kept; the values survive only in a note recorded from that output
(`MANIFEST.md` §8). That is the one gap in this ledger.

### 7.3 Capacity and weight averaging

| printed | source |
|---|---|
| 34.9306 M for the block-free, width-compensated instantiation | arm **M0**, `--no-mamba-bottleneck --fcc-channel 80`, defined in a sweep cell of the development repository (not shipped); `scripts/verify_release.py` §G checks it at 34,930,646 and `configs/ablation_commands.sh` documents it. **Never trained.** The paper states it as a parameter count only, and says so. |
| final-epoch checkpoint means .8701 | `run_final_2.txt:1107`, epoch 160 five-set mean |
| best of the eight snapshots on the in-domain splits means .8753 | `run_final_2.txt:799`, epoch 80 |
| their average, the shipped model, means .8833 | `runs/fin2_soup8_scored.json`, five-set mean of the §1 table |

**Why .8753 is legal.** Over all 32 monitored epochs the in-domain argmax is epoch 150, whose
five-set mean is .8642, and .8753 is simultaneously the maximum of the five-set column, which
looks like test selection. It is not. The soup's candidate pool is the eight cycle-end snapshots
only (`restart_cycle_len=20`), whose in-domain scores are 0.9234 / 0.9224 / 0.9211 / **0.9242** /
0.9158 / 0.9228 / 0.9132 / 0.9196. Epoch 80 is the in-domain argmax of that pool. The wording at
`sec/6_ablations.tex:83` was changed to "the best of the eight snapshots on the in-domain splits"
so the selection rule cannot be misread; the final paper says "the best single snapshot under the
in-domain rule".

---

## 8. Limitations and conclusion — `sec/8_conclusion.tex`

These are the draft's lines. The final conclusion keeps 34.85 M, the 19.4 points, the
boundary-distance claim, the 0.0040 spread and the lower-bound MAC figure; it drops the five Dice
values, the CVC-300 figure and the eight-of-eleven count, and adds the 26.4 FPS figure (header).

| line | printed | source |
|---|---|---|
| 11-12 | 0.0040 same-config spread, larger than two of the three steps | §7.1 |
| 15 | .8980 against a best encoder-scale .909 | `tab:main`; .908 after the 2026-08-27 correction, and the final conclusion no longer prints it |
| 17-19 | the MAC figure is a lower bound; throughput measured on a datacentre GPU | §1 cost block; L4 at `run_final_2.txt:265` |
| 27 | .9274 .9386 .8980 .8308 .8219 | `runs/fin2_soup8_scored.json` |
| 28 | up to 19.4 Dice points over the split-defining baseline | PraNet ETIS +.1935 |
| 29 | best or second on boundary distance across all five | `tab:hd95` |
| 30-31 | eight of eleven, all eleven on Kvasir and CVC-ColonDB | §6.3 |

---

## 9. Supplement — `supplementary.tex`

`tab:supp_profile` (all six threshold-swept measures) and `tab:supp_fixed` (Recall and HD95) are
`runs/fin2_soup8_scored.json` verbatim; the §1 table above is the same data. Image counts in
parentheses are the `n_images` field.

`fig3_degradation.pdf` plots the eleven re-scored methods from the same JSONs as `tab:hd95`.

`tab:supp_psrc` is `docs/analysis/psrc_scorecard.md`, per §6.6.

The sentence at `:105` about CVC-ColonDB's 380 images against CVC-300's 60 is the `n_images` field
of the two splits.

The final supplement adds two passages. Its layer shapes and numerical safeguards are read off
`src/model/net.py` and `src/model/fcc_modules.py`. Its failure gallery counts the frames below 0.1
per-image Dice: 1 of 100 on Kvasir, 8 of 380 on CVC-ColonDB, 10 of 196 on ETIS-Larib, and none on
CVC-ClinicDB or CVC-300. The counts are the `dice` fields of each split's `per_image` entries in
`results/fin2_soup8_scored.json`, recounted on 2026-11-02.

---

## 10. Fidelity notes: places the paper simplifies the code

None of these is wrong. They are recorded so they are not discovered under review.

1. **`eq:fuse` (`3_method.tex:51`)** writes `Conv_{1×1}` alone; `src/model/net.py:163-166` follows
   that convolution with a `BatchNorm2d`. The zero-init identity argument is unaffected, since the
   LayerScale gate γ_f is what carries it.
2. **The rFFT normalization is not stated.** `net.py:302` and `:336` use `norm='ortho'` on both
   `rfft2` and `irfft2`. The paper's equations are normalization-agnostic as written.
3. **The M5/M3 control pair is not literally the shipped configuration.** Both arms carry
   `--freq-gain-shared` and the bounded gain, which is why their parameter count is 34.8337 M and
   not the shipped 34.854150 M. §3.3 and §6.2 never claim the control arm is the shipped model,
   but neither says otherwise. Two distinct parameter counts appear in the paper for this reason
   and they must not be interchanged.
4. **`ablation_run_1.txt` (not shipped) is a different sweep.** It holds A0/A1/A2, the mixer
   knockout, and none of its numbers is printed in the paper: A1 (bi-axial) .8732 at `:374`, A0
   (tri-axial) .8689 at `:721`, A2 (no block) .8721 at `:1070`; ETIS Recall ladder .8976 → .9141
   → .9173; HD95 CVC-300 18.20 against 23.16 / 22.44 and ColonDB 45.87 against 51.53 / 48.52, with
   ETIS 123.90 the worst of the three. If any of this is ever added to the paper, note that A0
   loses all five datasets to A1, which is precisely why no accuracy claim is made for the
   spectral axis anywhere.
