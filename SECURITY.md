# Security policy

## Supported versions

The `main` branch and the checkpoint on the `fin2-weights` release.

## Checkpoints run code when loaded

`fin2_soup8.ckpt` is a PyTorch Lightning checkpoint, and Lightning checkpoints are Python
pickles. This repository loads them on purpose with `SpectraMambaLightning.load_from_checkpoint`
(`src/test.py`, `src/soup.py` and others) and with `torch.load(..., weights_only=False)`
(`src/soup.py`, `scripts/verify_release.py`, `scripts/verify_mamba_ref.py`). Both unpickle
arbitrary Python objects, so loading a checkpoint runs whatever code is inside it.

- Load only the checkpoint from this repository's `fin2-weights` release, and check its hash
  before anything reads it:

  ```bash
  echo "598de21496dbc7f7c0619ea0d2401c7dc28b6d5885c638359d695e0c69401e0b  fin2_soup8.ckpt" | sha256sum -c -
  ```

  `scripts/verify_release.py` reports a wrong hash as a failed check but then loads the file
  anyway, so it does not replace this step. The CI workflow runs the same `sha256sum -c`
  before any script touches the file.
- Never point this code at a checkpoint from a source you do not trust, including one
  presented as a fine-tuned or converted copy of this model.
- A newer Lightning does not remove the risk. CVE-2026-31221 (CWE-502, CVSS 3.1 score 7.8)
  describes the unpickling in `load_from_checkpoint` for pytorch-lightning 2.6.0 and earlier,
  and its advisory lists no fixed version. The published run used 2.6.5.

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private vulnerability reporting instead: open the
**Security** tab of this repository and choose **Report a vulnerability**. The code, the
workflows in `.github/workflows/` and the release assets are all in scope.

The repository is maintained by its authors on a best-effort basis.
