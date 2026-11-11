# Number Audit — every printed value, traced to primary evidence

Written 2026-08-13 against that day's draft of the paper, and edited on 2026-11-02 for this
repository. This is the **verification** layer under [PROVENANCE.md](PROVENANCE.md), whose
first table lists what the paper changed after the draft; the sections below note those
changes where they touch a check. Line numbers refer to the draft's LaTeX sources, which are
not in this repository.

`PROVENANCE.md` says where each number *is supposed to* come from. This file says what
happened when each of those claims was checked against the raw artefact, without
trusting `PROVENANCE.md`, without trusting the working transcription file
`literature_scores.md` (not shipped), and without trusting any earlier write-up.

Two things came out of it that change the paper. Both are fixed and recorded in §5.

| | outcome |
|---|---|
| **334** printed values machine-checked against their evidence | **0** failures |
| **16 / 16** `tab:main` competitor rows of the draft re-read against their source PDF | **0** transcription errors. The final paper's corrected IC-PolypSeg row and its new Polyp-DiFoM row are each read from the method's own paper (§3.1) |
| **5** source PDFs re-read at the table, three of them visually because the tables are rasterized | 3 inter-source conflicts found (§4) |
| numbers whose only surviving evidence is a note rather than a log file | **10** (§7, the one real gap) |

---

## 0. How it was run

Nothing here needs a GPU. The audit used three scripts and five source PDFs, none of which
ships with this repository: `verify_numbers.py` (104 checks over our own numbers, 104
passed), `verify_comp.py` (120 checks over competitor numbers, 120 passed) and `tab.py`,
which rebuilds a table row from a PDF page. The PDFs are the published papers and are not
redistributed; §3.1 names each table and page.

The shipped re-derivation is `scripts/verify_release.py` (`MANIFEST.md` §9). From the files
in this repository it re-checks the shipped model's scores, parameter counts and checkpoint
selection, the R0-R3 ablation, and all 396 paired-significance values. It does not re-read
competitor rows, which needs the source papers.

`tab.py` sorts PyMuPDF word boxes into rows by y-coordinate at a 3 pt tolerance. It
returns nothing on a rasterized table; three of the five source PDFs are rasterized at
the table (CTNet, and the wide landscape tables in CFA-Net and SAM-Mamba's Table 3),
and those were read visually instead. **A silent empty result from `tab.py` means "this
page has no extractable text", not "this row is absent"**; that trap cost one false
alarm during this audit.

---

## 1. The four evidence roots

Every number in the paper resolves to exactly one of these. Nothing else is admissible.

| class | what it means | root artefact |
|---|---|---|
| **RUN** | printed by a GPU training/scoring job | `run_final_2.txt`, `ABLATION_sweep.txt`, shipped as `logs/run_final_2.txt` and `logs/ablation_R0-R3.txt` with line numbers unchanged |
| **SCORED** | our estimator over prediction maps, no training | `runs/fin2_soup8_scored.json`, `runs/significance_fin2.json`, `docs/analysis/*predmaps*.json`, shipped as `results/fin2_soup8_scored.json`, `results/significance_fin2.json` and `results/competitors/` |
| **MEASURED** | CPU profiler, no training | `docs/analysis/efficiency_measured.json`, shipped as `results/efficiency.json` (`PROVENANCE.md` gives the one record that differs), and `docs/analysis/code_audit.md` (not shipped) |
| **TRANSCRIBED** | copied from another paper's printed table | the published paper's PDF, cited by table and page below (not redistributed) |

Derived values (ratios, differences, means) are marked as such and their arithmetic is
reproduced in `verify_numbers.py` §H.

---

## 2. Our own numbers — 104 checks, all passing

`verify_numbers.py` reads the two result JSONs and asserts every value the paper prints
about our own model. It does not read `PROVENANCE.md`, so a wrong number in that ledger
cannot make a check pass.

| § | what is checked | count |
|---|---|---|
| A | the five headline Dice at 3 dp (`tab:main`) and 4 dp (prose, abstract, conclusion), plus the five `n_images` | 15 |
| B | the four metric ranges in `sec/5_results.tex:54-56` — min and max of each column | 8 |
| C | the `tab:hd95` **Ours** row, HD95 and Recall | 10 |
| D | all **55** `tab:sig` cells, and independently all **55** star placements against `p_holm < 0.05` | 55 + 55 |
| E | the two counting claims | 3 |
| F | the ETIS right-tail sentence (mean > 0 while median < 0) | — |
| G | the five PraNet deltas in the abstract | 5 |
| H | every derived value in prose (leads, shortfalls, ratios, the honest MAC total) | 7 |
| I | the shipped five-set mean 0.8833 | 1 |

### 2.1 The three claims worth stating individually

**Star placement: zero mismatches in 55 cells.** Recomputed from `p_holm` rather than
copied. Per-dataset significance counts are Kvasir 11/11, ClinicDB 8/11, CVC-300 8/11,
ColonDB 11/11, ETIS 9/11 — so "on Kvasir and CVC-ColonDB against all eleven" is exact
and does not under-claim a third dataset.

**"Eight of the eleven on all five datasets"** — recomputed by intersecting the five
per-dataset significance sets: exactly 8. The three that fall out are MSEG, Polyp-PVT
and CTNet, which is what `tab:sig`'s `sig.` column prints.

**The soup selection is legal, and this was the hardest thing in the file to prove.**
`sec/6_ablations.tex` prints 0.8753 as "the best of the eight snapshots on the in-domain
splits". Over all 32 monitored epochs the *global* in-domain argmax is epoch 150, whose
five-set mean is 0.8642 — while 0.8753 is simultaneously the maximum of the five-set
column, which is exactly what test-set selection would look like.

It is not. The candidate pool is the eight cycle-end snapshots only
(`restart_cycle_len=20`), i.e. epochs 20, 40, … 160. Those eight in-domain scores are
**not printed anywhere in the log** — they are derived. Parsing all 32 evaluation blocks
in `run_final_2.txt` and computing `mean(Kvasir mDice, ClinicDB mDice)` at blocks
4/8/12/16/20/24/28/32 reproduces them exactly:

```
ep  20  40  60  80 100 120 140 160
  .9234 .9224 .9211 .9242 .9158 .9228 .9132 .9196
                     ^^^^^ pool argmax = epoch 80
```

Epoch 80's five-set mean is 0.8753 (`run_final_2.txt:794`). The selection rule and the
printed number agree. A grep for these eight values in the log finds nothing, which is
what triggered the false alarm; they are a derived statistic, and this paragraph is the
only place their derivation is written down.

### 2.2 The shipped model and the cost measurement are the same model

`tab:cost` reports 34.85 M / 16.37 G. Those come from arm **R3** of `ABLATION_sweep.txt`
(`:1423-1425`), not from the `fin2` run that produced the shipped weights. That is only
legitimate if the two configurations are identical on every flag that moves parameters
or MACs.

Checked flag by flag: `fin2`'s `[run-config]` (`run_final_2.txt:459`) and R3's `[config]`
(`ABLATION_sweep.txt:1423`) agree on all **11** such flags. Four more
(`--freq-gain`, `--freq-gain-scale`, `--freq-gain-shared`, `--third-axis`) are absent
from fin2's print; each takes its `argparse` default in `src/train.py`, and each default
equals the value R3 prints. The cost row therefore describes the shipped model.

### 2.3 Scan tokens, from first principles

`sec/3_method.tex:63-66` prints 4 972 against 9 680, a 1.95× ratio, and 48 664 against
58 080 network-wide for a 16% reduction. All four derive from the two stage sizes:

```
spatial      44^2 + 88^2                     = 1936 + 7744 = 9680
half-spectrum 44*(44/2+1) + 88*(88/2+1)      = 1012 + 3960 = 4972
ratio                                         9680/4972 = 1.9469 -> 1.95x
control (3 spatial scans, bidirectional)      3*9680*2      = 58080
ours   (2 spatial + 1 spectral, bidirect.)    (2*9680+4972)*2 = 48664
reduction                                     1 - 48664/58080 = 16.21% -> 16%
```

The factor 2 is bidirectional scanning. The retracted **19.35%** figure is the same pair
read as the control's *excess* over ours (58080/48664 − 1), and it appears nowhere in the
paper.

---

## 3. Competitor numbers — every row re-read against a source PDF

### 3.1 `tab:main`, all sixteen rows of the draft

Each row was re-read from the PDF page named below on 2026-08-13. **All sixteen
reconcile digit for digit with what the draft prints.** The evidentiary tier is what
differs, and `tab:main`'s caption has been corrected to say so (§5.1). Two rows below are
given as the final paper prints them: IC-PolypSeg, re-read from its own preprint on
2026-08-27 (§4.3), and Polyp-DiFoM, a row the final table adds.

| row | Kva / Cli / 300 / Col / ETIS | source table re-read | tier |
|---|---|---|---|
| U-Net | .818 .823 .710 .512 .398 | Dutta T1 p6, T2 p7, T3 p7 | secondary + map-corroborated¹ |
| U-Net++ | .821 .794 .707 .483 .401 | Dutta T1/T2/T3 | secondary + map-corroborated |
| PraNet | .898 .899 .871 .709 .628 | Dutta T1/T2/T3 | secondary + map-corroborated |
| SANet | .904 .916 .888 .753 .750 | Dutta T1/T2/T3 | secondary + map-corroborated |
| MSNet | .907 .921 .869 .755 .719 | Dutta T1/T2/T3 | secondary only¹ |
| HSNet | .926 .937 .903 .810 .808 | HAT-SAM3 T1 p7 | secondary only |
| M²SNet | .912 .922 .903 .758 .749 | Dutta T1/T2/T3 | secondary only |
| PVT-Cascade | .911 .919 .892 .781 .786 | Dutta T1/T2/T3 | secondary only |
| CFA-Net | .915 .933 .893 .743 .732 | **CFA-Net's own T1/T2 p7, T3 p8** | **primary** |
| Polyp-PVT | .917 .937 .900 .808 .787 | Dutta T1/T2/T3 | secondary + map-corroborated |
| MEGANet | .913 .938 .899 .793 .739 | Dutta T1/T2/T3 | secondary only |
| EMCAD | .894 .944 .878 .760 .744 | Agnihotri T1 p6, T2 p6-7 | secondary only |
| IC-PolypSeg | .910 .938 .907 .799 .758 | **own arXiv 2309.08234 T1-2, EF7 row** | **primary** (corrected 2026-08-27; the draft's .910 .938 .909 .807 .774 was HAT-SAM3 T1 p7, secondary only) |
| CTNet | .917 .936 .908 .813 .810 | Dutta; **also CTNet's own T III/IV pp8-9** | **primary** |
| Polyp-DiFoM | .909 .958 .889 .776 .759 | **Agnihotri T1-2 p6, EMCAD-backbone row** | **primary** (final paper only) |
| SAM-Mamba | .924 .942 .920 .853 .848 | **SAM-Mamba's own T1 p6, T2 p7, T3 p7** | **primary** |
| HAT-SAM3 | .929 .896 .929 .849 .815 | **HAT-SAM3's own T1 p7** | **primary** |

¹ *map-corroborated* = we hold that method's released prediction maps and our own
re-scoring of them agrees with the printed row. See §3.2.

Source PDFs, by the filename they were audited under (not redistributed):

- **Dutta** = `Dutta_SAM-Mamba_..._WACV_2025_paper.pdf`. Table 1 p6 (Kvasir, ClinicDB),
  Table 2 p7 (CVC-300, ColonDB), Table 3 p7 (ETIS). Supplies **12 of 16 rows**.
- **HAT-SAM3** = `s00371-026-04562-3.pdf`, Table 1 p7. Column order there is
  Kvasir / ClinicDB / **ColonDB / ETIS / CVC-300** / avg, not the paper's order; the
  three rows taken from it were re-checked against that ordering.
- **Agnihotri** = `Agnihotri_From_SAM_to_DINOv2_..._WACV_2026_paper.pdf`, Tables 1-2 p6-7,
  the Polyp-DiFoM paper.
- **IC-PolypSeg** = arXiv 2309.08234, Tables 1-2, read 2026-08-27 for the corrected row.
- **CFA-Net** = `1-s2.0-S0031320323002558-main.pdf` (Pattern Recognition 140:109555),
  Tables 1-2 p7, Table 3 p8, Table 4 p10. Rasterized — read visually.
- **CTNet** = `CTNet_Contrastive_Transformer_Network_for_Polyp_Segmentation.pdf`,
  Tables III-IV pp8-9. Rasterized — read visually.

### 3.2 Independent corroboration: transcription against our own measurement

For six methods we hold released prediction maps, so the printed row can be checked
against a measurement that shares nothing with it. `verify_comp.py` §L:

- **26 of 30** cells agree to within 0.0005 (their own rounding).
- **29 of 30** agree to within 0.0026.
- **1 outlier**: U-Net CVC-ColonDB, printed **0.512**, measured **0.5038** (Δ 0.0082).

That outlier is a real inter-source conflict, resolved in §4.1.

This is also the check that validates the estimator itself: CTNet's five printed cells
reproduce on our scorer to a maximum absolute error of 0.0005, which is the sentence at
`sec/4_experiments.tex:49-50`.

### 3.3 `tab:hd95` and `tab:sig` — nothing is transcribed

All 110 competitor cells in `tab:hd95` and all 55 in `tab:sig` are **SCORED** by us from
released maps; none appears in any published paper, because none of those papers reports
HD95 or Recall. `verify_comp.py` §J re-reads all 110 from the eleven per-method JSONs:
110 passed, 0 failed. This table has no transcription exposure at all.

---

## 4. Conflicts between sources, and how each was resolved

These are the reason a "the numbers all match" report would have been misleading. All
three are between *other people's* tables; none is an error in our transcription.

### 4.1 U-Net on CVC-ColonDB — 0.512 against 0.504

| source | value |
|---|---|
| Dutta (SAM-Mamba, WACV'25) Table 2 p7 — **what `tab:main` prints** | **0.512** |
| CFA-Net (PR'23) Table 2 p7 | 0.504 |
| HAT-SAM3 (2026) Table 1 p7 | 0.504 |
| our own re-scoring of U-Net's released maps | 0.5038 |

Three independent sources say 0.504; Dutta alone says 0.512. U-Net has no originating
number here — the 2015 paper predates these benchmarks — so there is no primary source
to appeal to, only a majority.

**Decision: keep 0.512.** The row is transcribed from Dutta, the caption now says the
rows come from published tables and names Dutta as the source of twelve of them, and
switching one cell to a different source would make the U-Net row inconsistent with the
eleven others beside it. Nothing in the paper's prose, ranking, bolding or significance
testing touches this cell — U-Net is the weakest row in the table, and every claim
against U-Net is made per image from its maps in `tab:sig`, not from this number.

The same pattern, smaller, on the same column: U-Net++ 0.483 (Dutta) against 0.482
(CFA-Net, HAT-SAM3) and 0.4815 (ours); PraNet 0.709 (Dutta, HAT-SAM3) against 0.712
(CFA-Net) and 0.7116 (ours).

### 4.2 MSNet — three sources, three different rows

| source | Kva / Cli / 300 / Col / ETIS |
|---|---|
| Dutta — **what `tab:main` prints** | .907 .921 .869 .755 .719 |
| CFA-Net (PR'23) | .905 .918 .865 .751 .723 |
| HAT-SAM3 (2026) | .905 .918 .862 .747 .720 |

Up to 0.008 apart. We hold no MSNet maps, so this cannot be adjudicated by measurement.
Kept as transcribed from Dutta, consistent with the rest of the table. MSNet is not
bolded, not underlined, and is named in no sentence.

### 4.3 Polyp-PVT ClinicDB — a propagated error we correctly avoided

HAT-SAM3's Table 1 prints Polyp-PVT ClinicDB as **0.948**. So does WaveSeq. The true
value is **0.937** (Dutta, Agnihotri, `2604.17865`, and Polyp-PVT's own paper), and our
re-scoring of Polyp-PVT's released maps gives **0.9368**.

`tab:main` prints 0.937. Two 2026 papers carrying the same wrong value is the
"transcription errors exceed claimed margins" finding of the cited 2026 benchmark audit
happening in our own bibliography, and it is the strongest argument for the §3.2
corroboration pass existing at all. It is also why HSNet and IC-PolypSeg, the two rows the
draft took from HAT-SAM3, were marked *secondary only* rather than trusted.

Vindicated 2026-08-27: the IC-PolypSeg preprint (arXiv 2309.08234) shows that the secondary
sources' .910/.938/.909/.807/.774 row is a per-dataset maximum over eight EfficientNet
variants (EF7/EF7/EF2/EF6/EF4). The paper now prints the single-model EF7 row,
.910/.938/.907/.799/.758, which moves the CVC-300 bold to CTNet's .908.

---

## 5. What changed in the paper because of this audit

### 5.1 `tab:main`'s caption was false and is fixed

It read:

> Competitor values are transcribed from the originating papers under the same split.

**Twelve of the sixteen rows are not from the originating paper.** They are from
SAM-Mamba's comparison table. Only CFA-Net, CTNet, SAM-Mamba and HAT-SAM3 are read from
the paper that produced the number. The draft's caption was changed to:

> Competitor values are transcribed from published tables on this split, twelve of the
> sixteen rows from one of them~\cite{dutta2025sammamba} so that those rows are mutually
> consistent.

The final caption reads "Competitor values are taken from published tables; the
Polyp-DiFoM row is its EMCAD-backbone configuration", and the experimental setup names
the two benchmark summaries the rows come from (Dutta and Agnihotri).

Single-source consistency is the defensible position, and it is what item C3 of the cited
benchmark audit's checklist (PSRC) asks for. The working transcription file,
`literature_scores.md`, carried the same claim in its header, against its own source
column, and was corrected with a dated note.

### 5.2 `tab:cost` now names its transcription source

`$^\dagger$ is transcribed` gave the reader no way to check five rows. All five, plus
SAM-Mamba's, resolve to **Agnihotri Table 1 p6**, and the caption now cites it. See §6.
The final table adds a second mark, ‡, for CTNet's row taken from CTNet's own paper (§6.1).

---

## 6. `tab:cost` — including the units question, which is the live risk

### 6.1 The five measured rows

`verify_comp.py` §K, against `docs/analysis/efficiency_measured.json`, shipped as
`results/efficiency.json` (**a list of records, not a dict**, a schema trap that crashed
the first run): Polyp-PVT 25.11/10.02, EMCAD 26.76/10.58, CFA-Net 25.83/55.36, PraNet
32.55/13.15, CTNet 44.29/15.14. All 10 values check out.

The final paper prints CTNet as 44.19‡/15.20‡ from CTNet's own Table II instead of our
measurement, which stays in `results/efficiency.json`, and adds a Polyp-DiFoM row that
repeats the EMCAD measurement, since Polyp-DiFoM ships in the EMCAD architecture.

The convention (`thop` MACs, 352², batch 1) rests on four external anchors, one exact:
**our measured CFA-Net 55.36 equals CFA-Net's own published Table 4 value 55.36**, and
their Table 4 also prints PraNet 13.15 against our 13.15. Agnihotri gives Polyp-PVT 10.1
(ours 10.02) and PraNet 13.1. Parameters run 0.6-2.0 M above the published counts on the
same models (CFA-Net 25.83 vs their 25.24, PraNet 32.55 vs their 30.50), most likely a
differently-counted Res2Net stem; we print our own measurement consistently.

### 6.2 The six transcribed rows, and why the column header cannot be trusted

All six — SANet 23.8/11.3, MSNet 27.6/17.0, M²SNet 27.7/17.1, PVT-Cascade 35.2/32.5,
MEGANet 44.1/28.8, SAM-Mamba 103.0/423.0 — are **Agnihotri Table 1 p6**, whose column
header reads `Params (M)  FLOPs (G)`.

That header is not reliable. Compare our own MAC measurements of rows in the same column:

| row | Agnihotri | our thop MACs | ratio |
|---|---|---|---|
| Polyp-PVT | 10.1 | 10.02 | 1.01 — MACs |
| PraNet | 13.1 | 13.15 | 1.00 — MACs |
| CFA-Net | 55.3 | 55.36 | 1.00 — MACs |
| EMCAD | 21.5 | 10.58 | **2.03 — FLOPs** |
| CTNet | 32.6 | 15.14 | **2.15 — FLOPs** |

**The column mixes conventions row by row.** The paper does not print either poisoned
cell — EMCAD and CTNet are among the rows we measured ourselves — but it means the six
transcribed rows carry an unresolved 2× risk, and only one of them is corroborated:
MSNet 17.0 matches CFA-Net's own Table 4 value 17.00, and CFA-Net's table is confirmed
MAC-convention by the two anchors above.

### 6.3 SAM-Mamba's 423.0 — the one transcribed number a claim rests on

`sec/5_results.tex:66` says the foundation-scale group costs "three times our parameter
count and over twenty times our compute". Parameters are safe: 103.0/34.85 = 2.96.
Compute is not, if 423.0 is FLOPs: 423/18.77 = 22.5× as MACs, but 211.5/18.77 = 11.3×
as FLOPs, and the sentence fails.

An analytic check settles it. SAM-Mamba is 103.0 M parameters, i.e. a SAM ViT-B image
encoder (89.7 M) plus adapters, run at 1024². ViT-B/16 at 1024² is 4096 tokens, d=768,
12 layers, with windowed attention in 10 of them:

```
per layer, windowed:  qkv+proj 4d^2N = 9.7 G   attn (14x14 win) 1.2 G   MLP 8d^2N = 19.3 G  ~= 30.2 G
per layer, global:    qkv+proj 9.7 G           attn N^2d*2 = 25.8 G     MLP 19.3 G          ~= 54.8 G
total  10*30.2 + 2*54.8  ~= 412 GMACs
```

412 against a printed 423 is a match under the **MAC** reading; under the FLOP reading
the printed value would be ~211 GMACs, half the analytic estimate. The claim holds, and
the paper's phrasing "over twenty times" is the form that survives either MAC convention
(traced 16.37 → 25.8×, honest 18.77 → 22.5×), which is why it is written that way.

**Superseded 2026-08-27.** The final paper claims only "an order of magnitude more
compute", which holds under either reading of 423.0 (25.8× as MACs, 12.9× if it is
FLOPs), and its cost table no longer prints 18.77.

---

## 7. The one real gap: ten numbers whose only evidence is a note

`sec/6_ablations.tex:59-63` and `sec/1_intro.tex:68` print the third-axis control:

```
spectral arm (M5)  .9281 .9314 .8890 .8195 .7954   mean .8727
spatial control(M3) .9177 .9223 .8921 .8210 .8170  mean .8740
```

**These two arms have no raw log in this repository.** They ran in a separate session
whose console output was not kept, and their scores survive only in a note written when
the arms finished. Every other number in the paper resolves to a file; these ten do not.
The derived 0.0013 difference, the "spectral leads in-domain, control leads on all three
unseen centres" sentence, and the intro's framing of the third axis as a token-economy
rather than accuracy argument all rest on them.

What *is* independently verifiable about that control is its legitimacy as a control:
`--third-axis raster` has a parameter delta of exactly **0.000000**
(`docs/analysis/code_audit.md:378-388`, not shipped), so "params-matched at 34.8337 M" is
measured, not asserted. `scripts/verify_release.py` §G rebuilds both arms and checks that
each counts 34,833,672 parameters, and `configs/ablation_commands.sh` gives the commands
that train them.

This is the answer to "if a reviewer asks for one log, which one is missing".

---

## 8. Standing traps this audit hit

Recorded so the next pass does not lose time to them.

1. **`efficiency_measured.json` is a list**, not a dict. `json.load(...).keys()` raises.
   Rebuild as `{r['model']: r for r in raw}`.
2. **The development copy of that file also carried an `"ours"` entry at 35.315142 M /
   15.679 G**, the model at `train.py`'s argparse defaults rather than the shipped flag set.
   It describes no shipped model; `results/efficiency.json` replaces it with the shipped
   model's record, and `results/COST.md` explains the change.
3. **`tab.py` returning nothing means the page is rasterized**, not that the row is
   missing. Check with a count of `0.\d{3}` matches per page before concluding anything.
4. **HAT-SAM3's column order is Kva/Cli/Col/ETIS/300**, not the paper's order. Reading it
   positionally without checking the header silently swaps CVC-300 and CVC-ColonDB.
5. **Write the script to a file.** A regex containing `\label` or `\hbox`, inlined into
   `python -c` through a shell, dies as `re.error: bad escape`. On Windows, every script
   that prints repository text also needs
   `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`.
6. **The eight soup-pool in-domain scores are derived, not logged.** Grepping for them
   finds coincidental digits in unrelated rows. Compute them from the 32 evaluation
   blocks (§2.1).

---

## 9. Verdict

Every number the paper prints is traceable to a primary artefact, and 334 of them were
re-derived from that artefact during this audit without a single mismatch. The three
things that were *not* right were all provenance claims rather than values: `tab:main`'s
caption, `tab:cost`'s unattributed dagger, and `literature_scores.md`'s header. All three
are corrected. The residual exposure is ten ablation values that live in a note instead of
a log, and six transcribed cost cells whose source column mixes MAC and FLOP conventions.
Both are stated here, and the second does not affect any claim the paper makes.
