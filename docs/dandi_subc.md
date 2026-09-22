# DANDI688 Sub-C training reproduction

This release is SUA-only. Set `APST_DATA_ROOT` to the directory that contains
DANDI 000688 Sub-C NWB files, normally `$APST_DATA_ROOT/000688/sub-C` if the
DANDI download root is `$APST_DATA_ROOT`. Set `APST_SUBC_ROOT` to an empty
artifact root. No template is frozen directly: `init-config` writes its local
cache receipt binding and adjacent SHA sidecar first.

```bash
export APST_DATA_ROOT=/data/000688/sub-C
export APST_SUBC_ROOT=$PWD/runs/dandi_subc
python -m apst.dandi_subc prepare --cache "$APST_SUBC_ROOT/prepared_sua"

for name in sua_activity_pretrain_f0 sua_concat_pretrain_f0 sua_full_f0 sua_act_f0 sua_e0only_f0 sua_tokenonly_f0 sua_full_flat; do
  python -m apst.dandi_subc init-config --template "examples/dandi_subc/${name}.json" --cache "$APST_SUBC_ROOT/prepared_sua" --dest "$APST_SUBC_ROOT/configs/${name}.json"
done

python -m apst.dandi_subc pretrain --config "$APST_SUBC_ROOT/configs/sua_activity_pretrain_f0.json" --dest "$APST_SUBC_ROOT/pretrain_activity" --device cuda:0
python -m apst.dandi_subc pretrain --config "$APST_SUBC_ROOT/configs/sua_concat_pretrain_f0.json" --dest "$APST_SUBC_ROOT/pretrain_concat" --device cuda:0

python -m apst.dandi_subc stage2 --config "$APST_SUBC_ROOT/configs/sua_act_f0.json" --encoder "$APST_SUBC_ROOT/pretrain_activity/encoder.pt" --dest "$APST_SUBC_ROOT/act_f0" --device cuda:0
python -m apst.dandi_subc stage2 --config "$APST_SUBC_ROOT/configs/sua_tokenonly_f0.json" --encoder "$APST_SUBC_ROOT/pretrain_activity/encoder.pt" --dest "$APST_SUBC_ROOT/tokenonly_f0" --device cuda:0
python -m apst.dandi_subc stage2 --config "$APST_SUBC_ROOT/configs/sua_full_f0.json" --encoder "$APST_SUBC_ROOT/pretrain_concat/encoder.pt" --dest "$APST_SUBC_ROOT/full_f0" --device cuda:0
python -m apst.dandi_subc stage2 --config "$APST_SUBC_ROOT/configs/sua_e0only_f0.json" --encoder "$APST_SUBC_ROOT/pretrain_concat/encoder.pt" --dest "$APST_SUBC_ROOT/e0only_f0" --device cuda:0
python -m apst.dandi_subc stage2 --config "$APST_SUBC_ROOT/configs/sua_full_flat.json" --encoder "$APST_SUBC_ROOT/pretrain_concat/encoder.pt" --dest "$APST_SUBC_ROOT/full_flat" --device cuda:0
```

The activity pretrain is shared by ACT and token-only; concat pretrain is
shared by Full and E0-only. Stage 2 preserves source-only fitting and freezes
the corresponding source encoder trunk. Full uses profile shuffle only as a
final condition; it is not a development selection condition. F0/flat is
restricted to Full. The full-clock output EMA uses `alpha=1/3` as previous-state
coefficient: `p <- p/3 + 2*row/3`, reset per session before Q50 gathering.

Continue with [Static and final scoring](dandi_subc_final.md), the
[calibration-budget curve](dandi_subc_budget.md), and [WF controls](dandi_subc_baselines.md).
WF consumes target-local dense velocity labels on M33 and is not an APST/ACT
no-target-update comparator.
