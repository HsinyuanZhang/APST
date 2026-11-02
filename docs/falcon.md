# FALCON M1/M2/H1 training

This module trains APST decoders for the three FALCON tasks from the public
DANDI recordings: two stages of gradient training on held-in source
sessions, then closed-form, gradient-free calibration on each target
session. See `final_submissions/` for the submission artifacts and
streaming runtimes.

## Layout

- Stage 1 (`pretrain`) trains the association-profile trunk (component A)
  with a supervision decoder on held-in source recordings.
- Stage 2 (`stage2`) freezes A, optionally trains the rank-8 carrier FiLM,
  and trains a fresh F0 decoder (noPE temporal with a ladder-scaled
  learned-slope initialization) over the frozen banks.
- `calibrate` exports per-session association-profile banks
  (`E0.npy`, `carrier.npy`, `unit_mask.npy`); a trained FiLM is baked into
  the exported E0.

One configuration schema covers all tasks. `film` toggles the carrier FiLM
and `ladder_scale` rescales the recency half-life ladder; the shipped
examples use per-task default scales (M1 16.0, M2 1.0, H1 2.0) with FiLM
on. `seed` is a free hyperparameter — nothing in the module is bound to a
particular seed.

## Workflow

```bash
export APST_DATA_ROOT=/path/to/data          # contains 000941/000953/000954
export APST_FALCON_ROOT=/path/to/runs

python -m apst.falcon prepare --cache runs/falcon/cache --tasks m1 m2 h1

python -m apst.falcon init-config \
  --template examples/falcon/m1_pretrain.json \
  --cache runs/falcon/cache --dest runs/falcon/m1_pretrain.json

python -m apst.falcon pretrain \
  --config runs/falcon/m1_pretrain.json --dest runs/falcon/m1_pre --device cuda

python -m apst.falcon init-config \
  --template examples/falcon/m1_stage2.json \
  --cache runs/falcon/cache --dest runs/falcon/m1_stage2.json

python -m apst.falcon stage2 \
  --config runs/falcon/m1_stage2.json \
  --encoder runs/falcon/m1_pre/epoch_009.pt \
  --dest runs/falcon/m1_stage2 --device cuda

python -m apst.falcon calibrate \
  --config runs/falcon/m1_stage2.json \
  --checkpoint runs/falcon/m1_stage2/epoch_002.pt \
  --dest runs/falcon/m1_banks
```

`--tasks m1 m2 h1` can be replaced by any subset. Every command prints a
JSON receipt; training runs additionally write `run_meta*.json`, per-epoch
checkpoints, and a final `train_receipt.json`/`score_receipt.json` in the
(fresh) destination directory.

## Selection and contracts

Development is the public held-out calibration surface only: M1 uses three
held-out sessions (equal-session mean), M2 six held-out sessions
(equal-session mean), and H1 the fourteen held-out recordings scored
grouped-seven (S6–S12). Stage-2 selection runs the inference contract —
causal output EMA `alpha = 1/3` over the full recording clock, earliest
finite maximum. Target-session parameters are never updated: calibration
is one frozen `no_grad` forward pass.

## Smoke run

`--smoke-updates N` (1–8) runs a single-update training cycle, a truncated
evaluation, and the full selection/replay/bank machinery, writing
`smoke_receipt.json` instead of a formal receipt:

```bash
python -m apst.falcon pretrain --config ... --dest ... --device cpu --smoke-updates 1
```

See [the association-profile construction guide](falcon_profiles.md) for
the per-task carrier estimators and closed-form calibration formulas.
