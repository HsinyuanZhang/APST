# Sub-C final scoring

Complete the source/development runs in [the training guide](dandi_subc.md)
first. Train the Static-F0 development-selected control before opening final
data; select the WF controls with [the baseline guide](dandi_subc_baselines.md).

```bash
python -m apst.dandi_subc static \
  --cache "$APST_SUBC_ROOT/prepared_sua" \
  --dest "$APST_SUBC_ROOT/static" --device cuda:0
```

Seal the selected SUA checkpoints. The seal binds each selected checkpoint,
source statistics, selection, and training receipt. All selections use only
development recordings.

```bash
python -m apst.dandi_subc.finalize seal \
  --dest "$APST_SUBC_ROOT/final/selection_seal.json" \
  --run "full_f0=$APST_SUBC_ROOT/full_f0" \
  --run "act_f0=$APST_SUBC_ROOT/act_f0" \
  --run "e0only_f0=$APST_SUBC_ROOT/e0only_f0" \
  --run "tokenonly_f0=$APST_SUBC_ROOT/tokenonly_f0" \
  --run "full_flat=$APST_SUBC_ROOT/full_flat"
python -m apst.dandi_subc.finalize score \
  --seal "$APST_SUBC_ROOT/final/selection_seal.json" \
  --dest "$APST_SUBC_ROOT/final/core" --device cuda:0
```

If reproducing only Full and ACT, omit the other `--run` arguments. The score
command evaluates the six final recordings and includes profile shuffle for
Full. It creates `final/core/final_cache/` and the corresponding
`final_cache_receipt.json` for subsequent fixed-checkpoint comparisons.

```bash
python -m apst.dandi_subc.static_eval \
  --seal "$APST_SUBC_ROOT/final/selection_seal.json" \
  --final-cache-receipt "$APST_SUBC_ROOT/final/core/final_cache_receipt.json" \
  --static-run "$APST_SUBC_ROOT/static" \
  --dest "$APST_SUBC_ROOT/final/static" --device cuda:0
```

Continue with the [4/8/16/32 budget evaluation](dandi_subc_budget.md) and
[WF final evaluation](dandi_subc_baselines.md). These use the same sealed cache
and fixed development selections. Source statistics are not refitted.
