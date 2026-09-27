# DANDI688 release validation

The release is a portability extraction of the recorded Sub-C and Sub-M
experiments. Historical results are listed separately in
[`results/dandi688_reference.json`](../results/dandi688_reference.json).

The following checks use CPU and do not rerun the full training campaigns:

- For each subject, compare ACT-only, E0-only, Token-only, Full F0, and Full flat
  against the original experiment model with the same initialization and
  generated calibration/query inputs. Every state tensor matches exactly;
  the maximum prediction difference is zero in all ten comparisons.
- Check that ACT-only rejects a supplied association profile.
- Check that stage-two training freezes the source encoder while the
  zero-initialized FiLM adapter starts at the original encoder output and
  receives gradients.
- Check the historical inference EMA convention: the previous-state
  coefficient is 1/3, so `[0, 3, 0]` becomes `[0, 2, 2/3]`. State starts from
  the first prediction on each recording. EMA is not applied to training loss.
- Check that optimizer groups exclude frozen parameters and give
  `slope_log` zero weight decay.

Run the public contract checks with:

```bash
python -m pip install -e ".[experiments,test]"
python -m pytest -q tests/test_dandi_release.py
```

Additional extraction checks completed before publication:

- Materialize and reload one public Sub-C source NWB (74 sorted M1 units).
- Run Sub-C CPU source pretraining and encoder handoff with one update per
  stage, marked `SMOKE`, with no final sessions opened.
- Run Sub-M CPU source pretraining, encoder handoff, development selection,
  checkpoint reload, and development replay with two updates per stage on a
  bounded 128-bin fixture derived only from source/development recordings.
  These are mechanism checks, not performance estimates.
- Exercise all eight Full/ACT × calibration-budget paths on a small synthetic
  recording and check fixed source normalization and ACT profile exclusion.
- Round-trip a selection seal and reject a checkpoint changed after sealing.

Full reproduction requires downloading the public NWB files and following
both subject guides. Published checkpoints and raw data are not part of this
source release.

## September 2026 portability update

The APST entrypoints were rechecked for both historical rosters: Sub-C
18/6/6 and Sub-M 6/2/3 source/development/final sessions. The Sub-C default
artifact directory now follows the working directory instead of the installed
package location. Both subject guides install the experiment dependencies and
specify the directory containing raw NWB files. Sub-M preparation requires a
fresh destination and its configuration generator requires a preparation receipt.

CPU checks cover the existing model contracts and Full/ACT support construction
at 4, 8, 16, and 32 trials. Final APST, Static, WF, and budget entrypoints use
artifacts generated under the new run directory; the historical campaign is not
an input dependency.

The RNN extraction was compared directly with the original campaign on a
seed-42 synthetic 100-slot recording. Causal windows and label normalization
were identical. Source training (two epochs with two updates each), target
training from scratch (four updates), and source-pretrained fine-tuning (four
updates) produced exactly equal parameter tensors. Full-recording FP64 EMA,
variance-weighted R², and per-output R² also matched exactly. These CPU checks
verify the extracted computation without rerunning the full training campaigns.

The RNN support adapter was also checked on the original public source
recordings `sub-C_ses-CO-20150309` and `sub-M_ses-CO-20150511`.
For each recording, all 4/8/16/32-trial endpoint arrays matched the original
budget implementation exactly; M32 also matched the original cached support.
No final recordings were scored during these checks.

The combined CPU test suite passes 12 tests, including reduced RNN training,
selection and final evaluation, seal validation for both subjects, and lazy
final materialization after development selection. The package builds as a
wheel; installation and CLI import checks are performed outside the source
checkout.
