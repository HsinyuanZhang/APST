# DANDI 000688 recurrent baseline

`apst.dandi_rnn` is a public, CPU/GPU runnable one-layer LSTM comparison for the
Sub-C and Sub-M 2015 SUA rosters. It has four separately reported outputs:
source pretraining, source-pretrained zero-shot inference, target-session
training from scratch, and source-pretrained full-parameter target fine-tuning.
The recurrent architecture follows the movement-decoding baseline in
[Karpowicz et al., FALCON (NeurIPS 2024), Appendix A.4.2](https://papers.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf).
Its [reference implementation](https://github.com/joel99/context_general_bci/blob/main/context_general_bci/model.py)
and [hyperparameter definitions](https://github.com/joel99/context_general_bci/blob/main/context_general_bci/config/hp_sweep_space.py)
provide the FALCON baseline context. The code here implements the DANDI experiment:
20-ms sorted SUA, 100 right-zero-padded slots, a 50-bin causal history,
a two-dimensional velocity readout, and the splits and calibration rules below.

Install the experiment dependencies, then materialize the standalone portable
cache directly from public NWB. The adapter retains the original
Sub-C 18/6/6 and Sub-M 6/2/3 rosters and derives each M=4/8/16/32 support from
the first M legal rewarded trials and complete causal histories.

```bash
python -m pip install -e '.[experiments]'
python -m apst.dandi_rnn.adapter --subject C --raw-root data/000688/sub-C \
  --out runs/dandi_rnn_c_records
python -m apst.dandi_rnn --manifest runs/dandi_rnn_c_records/manifest.json \
  --config configs/dandi_rnn/default.json --out runs/dandi_rnn_c --raw-root data/000688/sub-C --device cuda:0
```

`manifest.json` records relative NPZ paths and optional digests; every portable
NPZ stores `neural`, `velocity`, Q50 `query`, and `support_4/8/16/32` endpoint
arrays. With `--raw-root`, the runner automatically materializes final records
after writing its development seal. It checks the protocol, roster, frozen
configuration and source/development manifest snapshots, and source-checkpoint
digest before final access.

For Sub-M, replace `C` with `M` and use `data/000688/sub-M` consistently:

```bash
python -m apst.dandi_rnn.adapter --subject M --raw-root data/000688/sub-M --out runs/dandi_rnn_m_records
python -m apst.dandi_rnn --manifest runs/dandi_rnn_m_records/manifest.json \
  --config configs/dandi_rnn/default.json --out runs/dandi_rnn_m --raw-root data/000688/sub-M --device cuda:0
```

The default preserves the campaign surfaces: source uses six H/LR cells,
20 epochs and 256 updates per epoch, source-session Q50 query labels,
and saves the earliest best dev epoch. Scratch selection is the six-cell M32
surface with 200 epochs, 16 updates per epoch, and the historical evaluation
epochs. Fine-tuning loads the selected source checkpoint and searches M32 only
over LR `1e-5,3e-5,1e-4,3e-4` and updates
`0,16,32,64,128,256,512`. Both target arms freeze their M32 recipe and reuse it
unchanged at 4/8/16/32 trials. The common development seal is written before
the final manifest is loaded. Full-clock inference uses
`p[t] = p[t-1]/3 + 2*raw[t]/3`, reset for each session.

For a no-data CPU verification run:

```bash
pytest -q tests/test_dandi_rnn.py
```

This synthetic smoke executes source training, development selection, the
frozen seal, final zero-shot evaluation, scratch budgets, and fine-tuning
budgets with intentionally reduced settings. It does not establish scientific
performance or data-cache provenance.

Historical final means for zero-shot, target-from-scratch and source-pretrained
fine-tuning are in [the RNN reference ledger](../results/dandi_rnn_reference.json).
New run outputs are written to the specified `--out` directory:

- `source_development.json`: source candidate curves and selected checkpoint.
- `development_seal.json`: selected source and M32 target recipes.
- `frozen_config.json` and `frozen_source_dev_manifest.json`: input snapshots.
- `results.json`: zero-shot scores and 4/8/16/32-trial scratch/fine-tuning scores,
  with per-session R² and query, truth, and prediction hashes.
