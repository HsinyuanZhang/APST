# Sub-C calibration budget

The fixed selected Full and ACT checkpoints are evaluated at M=4,8,16,32 on
the sealed six-session final cache. M changes only activity E0 and, for Full,
the MOVE-T4 association profile; source statistics are never refit, query Q50
is unchanged, and prediction remains the original full-clock EMA path. A Full
session with direction-design rank below three is `UNAVAILABLE_RANK_LT3` and is
retained rather than omitted from an average.

```bash
python -m apst.dandi_subc.budget --seal $APST_SUBC_ROOT/final/selection_seal.json --final-cache $APST_SUBC_ROOT/final/core/final_cache --final-receipt $APST_SUBC_ROOT/final/core/final_cache_receipt.json --dest $APST_SUBC_ROOT/budget/final --device cuda:0
```
