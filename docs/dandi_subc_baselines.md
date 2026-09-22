# Sub-C target-local Wiener-filter baselines

`WF-FSS`, `PCA-WF`, and `FA-WF` are SUA target-local baselines. Every fitted
readout consumes dense velocity labels on the allowed M33 carrier window; they
are therefore not parameter-update-free comparators to APST/ACT. WF-FSS uses
its historical frozen configuration. PCA-WF and FA-WF select their frozen grid
on the six development recordings only; FA candidates whose requested rank is
not legal or whose sealed fitting retry fails are unavailable, never silently
rank-reduced.

```bash
python -m apst.dandi_subc.baselines select-dev --cache "$APST_SUBC_ROOT/prepared_sua" --dest "$APST_SUBC_ROOT/wf/selection.json"
python -m apst.dandi_subc.baselines score-final --seal "$APST_SUBC_ROOT/final/selection_seal.json" --final-cache "$APST_SUBC_ROOT/final/core/final_cache" --final-receipt "$APST_SUBC_ROOT/final/core/final_cache_receipt.json" --selection "$APST_SUBC_ROOT/wf/selection.json" --dest "$APST_SUBC_ROOT/wf/final.json"
```

Final scoring requires the neural final seal and its materialized final cache.
It does not open raw NWB data or reselect any PCA/FA candidate.
