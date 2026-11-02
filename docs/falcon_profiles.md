# Association-profile construction

The association profile is the target-session conditioning consumed by the
decoder: a per-unit vector `E0 [units, e0_dim]` plus a 4-D carrier
`[units, 4]`, frozen at calibration time. This page documents the
construction implemented in `apst.falcon.profile` (encoder),
`apst.falcon.carrier` (carrier estimators), and `apst.falcon.data`
(support construction).

## Support activity

Per task, the chronological support prefix of every recording is separated
into per-trial tensors and cubic-resampled to a fixed trial length:

| Task | Support | Per-trial tensor | Selection |
| --- | --- | --- | --- |
| M1 | first 10 trials | `[10, 1024, 64]` | trial starts from `trials.start_time` on the eval-mask timestamps; eval-valid bins |
| M2 | first 33 trials | `[33, 100, 96]` | trial-change boundaries of the recording prefix |
| H1 | first 3 eval-valid `TrialNum` trials | `[3, 1024, 176]` | `TrialNum` + eval mask |

Support builders never receive a query target. Query planes are padded
timelines (`context - 1` zero bins) whose window starts are raw endpoint
indices; source queries additionally start at/after the first
post-support trial.

## Profile encoder (component A)

For one session with support activity `a [M, B, N]` and carrier `c [N, 4]`:

```text
pooled = mean_over_trials( ReLU(Linear_{B->h}( a[:, :, n] )) )      # [N, h]
h'     = FiLM(pooled, c)           # optional, rank-8, zero-init => identity
E0     = MLP3( concat(h'_n, c_n) ) # [N, e0_dim]
```

Equal-weight pooling over calibration trials; the carrier enters detached.
Per-task sizes (`h`, `e0_dim`) are frozen in `protocol_manifest.json`
(M1 64/100, M2 64/50, H1 32/700). Calibration is closed form: with the
trunk frozen, `E0` is one `no_grad` forward pass. When FiLM is trained in
stage 2, the trained FiLM is baked offline into exported banks,
`E0 = MLP3(cat(FiLM(pooled, c), c))`.

## Carrier estimators (all closed form, source-frozen)

**M2 — MOVE-T4 direction OLS.** For each of the first 33 trials, spike
counts are summed over the movement window (bins 5–30 of the trial);
per unit, `[1, cos θ, sin θ]` is regressed on those sums with `θ` the
trial's target angle from `tgt_loc` (screen centre 0.5, 0.5; centre/rest
trials are unlabelled and skipped). Rows `[w_cos, w_sin, ||w||, baseline]`
are z-scored with a per-column mean/std fitted on held-in source rows
only. Rank-deficient designs are rejected (the estimator raises unless the
cosine design has full rank 3). This reuses
`apst.legacy_dandi.carrier.estimate_move_t4`.

**M1 — muscle-recruitment SVD4.** EMG is rectified and standardized by a
source RMS (fitted on source EMG through trial 309); rates are standardized
with the Poisson floor `sqrt(max(r̄·Δ, 1))/Δ` (`Δ = 0.02 s`). The
descriptive response per unit/muscle is
`Σ_t w_tm·z_tu / (Σ_t w_tm + 10)` with behavioural weights
`w = max(EMG, 0)/RMS`. Source-pooled unit rows are reduced by SVD to 4
components (deterministic sign), normalized with the source global-RMS
scale. The frozen `{RMS, V4, μ, s}` fit is applied unchanged to every
target M10.

**H1 — signed-state behavioral profile.** Support trials are cut into
100-ms blocks (5 native bins; rates = block sum / 0.1 s, velocities = block
mean, float64 aggregation). Rates use the same Poisson floor
`sqrt(max(mean count, 1))/0.1 s`; behaviour enters as smooth signed states
`softplus(±v/RMS)` (14 columns). The per-unit raw profile is the
occupancy-shrunk weighted mean (`+10` pseudo-blocks). A 4-D basis, mean,
and scale are fitted on the pooled source profiles and applied frozen to
target support.

## Calibration budgets

`calibrate` exports banks at each session's protocol budget (M1 10, M2 33,
H1 3). A budget curve is the same closed-form pass over a trial prefix
(`budget_slice`), never a refit of the source statistics.
