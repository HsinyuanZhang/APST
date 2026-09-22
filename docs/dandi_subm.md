# DANDI000688 Sub-M release workflow

This release reproduces the fixed 2015 M1-only sub-M campaign: six source
sessions, two development sessions, and a capability-gated three-session
final pass.  It does not contain cached data, checkpoints, prior receipts, or
final predictions.

Set the raw NWB directory and a fresh artifact directory before every command:

```bash
export APST_DATA_ROOT=/absolute/path/to/data/000688/sub-M
export APST_SUBM_ROOT=/absolute/path/to/subm_artifacts
export PYTHONNOUSERSITE=1
PY=python
```

Install the release into the active environment first (for example,
`pip install .`).  The commands do not require a checkout-specific
`PYTHONPATH`.

Materialize and audit the source/development SUA cache (the final roster is
only schema-audited at this stage):

```bash
$PY examples/dandi_subm/prepare.py
$PY examples/dandi_subm/make_configs.py
```

Train the frozen M1-only, 6/2/3 source/development matrix.  The two pretrains
must precede their respective train stages.  The train scripts retain the
24×3165 update schedule, full-recording EMA with alpha 1/3, zero slope weight
decay, and earliest-max development selection.

```bash
$PY examples/dandi_subm/train.py "$APST_SUBM_ROOT/configs/sua_activity_pretrain_s42.json" --device cuda:0
$PY examples/dandi_subm/train.py "$APST_SUBM_ROOT/configs/sua_act_only_F0_s42.json" --device cuda:0
$PY examples/dandi_subm/train.py "$APST_SUBM_ROOT/configs/sua_concat_pretrain_s42.json" --device cuda:0
$PY examples/dandi_subm/train.py "$APST_SUBM_ROOT/configs/sua_dual_site_F0_s42.json" --device cuda:0
```

Run the independent Static-F0 and WF baseline selection surfaces:

```bash
$PY examples/dandi_subm/static/run_static_source.py train --device cuda:0
$PY examples/dandi_subm/baselines/run_historical.py select
```

The calibration-budget evaluator uses fixed selected checkpoints and supports
budgets 4, 8, 16, and 32 for ACT-only and APST-FULL.  It uses development data
until the shared final seal and cache exist.

```bash
$PY examples/dandi_subm/budget/evaluate.py --split dev --device cuda:0
```

After all development selections are formal, seal once, score APST ACT/Full,
then score Static-F0, WF, and the final calibration-budget surface.  Each
final command requires the newly created seal/cache and cannot reuse private
campaign outputs.

```bash
$PY examples/dandi_subm/finalize.py seal
$PY examples/dandi_subm/finalize.py score --seal "$APST_SUBM_ROOT/final/selection_seal.json" --device cuda:0
$PY examples/dandi_subm/static/eval_final.py --seal "$APST_SUBM_ROOT/final/selection_seal.json" --final-cache "$APST_SUBM_ROOT/final/final_cache" --final-receipt "$APST_SUBM_ROOT/final/final_cache_receipt.json" --device cuda:0
$PY examples/dandi_subm/baselines/final_driver.py --seal "$APST_SUBM_ROOT/final/selection_seal.json" --final-cache "$APST_SUBM_ROOT/final/final_cache" --final-receipt "$APST_SUBM_ROOT/final/final_cache_receipt.json"
$PY examples/dandi_subm/budget/evaluate.py --split final --device cuda:0
```
