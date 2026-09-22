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
