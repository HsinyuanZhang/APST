# APST

Code for **APST** (association profile conditioning) on the
[FALCON](https://snel-repo.github.io/falcon) few-shot neural decoding
benchmark and on [DANDI 000688](https://dandiarchive.org/dandiset/000688/0.250122.1735)
SUA reaching recordings. The DANDI688 experiment code includes source pretraining,
frozen-encoder decoder training, development selection, held-out evaluation,
association-profile controls, and calibration-budget evaluation for **Sub-C**
and **Sub-M**. FALCON download and loading utilities remain available.

## Data

| Dataset | Source | Notes |
|---------|--------|-------|
| FALCON M1 | [DANDI 000941](https://dandiarchive.org/dandiset/000941) | `sub-MonkeyL` |
| FALCON M2 | [DANDI 000953](https://dandiarchive.org/dandiset/000953) | `sub-MonkeyN` |
| FALCON H1 | [DANDI 000954](https://dandiarchive.org/dandiset/000954) | `sub-HumanPitt` |
| DANDI 688 SUA | [000688 v0.250122.1735](https://dandiarchive.org/dandiset/000688/0.250122.1735) | `sub-C`, `sub-M`, `sub-J` |

FALCON dandisets have three public splits: `held-in-calib`, `held-in-minival`,
and `held-out-calib`. Official FALCON test labels stay on EvalAI.

DANDI 000688 is used as sorted SUA. Download only `sub-C` / `sub-M` / `sub-J`
(`sub-T` is threshold crossings and is not fetched).

### Install

```bash
python -m pip install -e .
python -m pip install "falcon-challenge @ git+https://github.com/snel-repo/falcon-challenge.git"
```

`dandi` and `pynwb` are enough to download everything and to load DANDI 000688.
`falcon-challenge` is required only to load FALCON NWB files with the official
evaluator reader.

### Download

```bash
# FALCON M1/M2/H1 and DANDI 000688 SUA into ./data/
python -m apst.data.download

# FALCON only
python -m apst.data.download --tasks m1 m2 h1

# DANDI 000688, one subject
python -m apst.data.download --tasks dandi688 --subjects sub-C
```

Set `APST_DATA_ROOT` (or `EVAL_DATA_PATH`) if the files are not under `./data`.

Equivalent DANDI CLI:

```bash
dandi download https://dandiarchive.org/dandiset/000941 -o data
dandi download https://dandiarchive.org/dandiset/000953 -o data
dandi download https://dandiarchive.org/dandiset/000954 -o data
dandi download "https://api.dandiarchive.org/api/dandisets/000688/versions/0.250122.1735/assets/?path=sub-C" -o data/000688
```

### Load FALCON

```python
from apst.data import list_sessions, load_session

rows = list_sessions("m1", split="held_in")
item = load_session("m1", session=rows[0]["session"], split="held_in")
neural = item["neural"]          # time x units
covariates = item["covariates"]  # time x outputs
eval_mask = item["eval_mask"]    # bins scored by FALCON
```

`load_session` / `load_nwb_file` call
`falcon_challenge.dataloaders.load_nwb`, so binning, EMG/kinematics, trial
boundaries, and `eval_mask` match the EvalAI evaluator.

### Load DANDI 000688

```python
from apst.data import list_dandi688_sessions, load_dandi688_session

rows = list_dandi688_sessions(subject="sub-C")
item = load_dandi688_session("sub-C_ses-CO-20150716")
neural = item["neural"]      # 20 ms M1 spike counts, time x units
velocity = item["velocity"]  # interpolated cursor velocity
trials = item["trials"]      # trial intervals from the NWB
```

## Acknowledgements

We thank the FALCON organizers for the benchmark, the public NWB layout, and
the evaluation protocol, and the EvalAI team for hosting the
[FALCON challenge](https://eval.ai/web/challenges/challenge-page/2319/overview).
Recordings are released on DANDI by the original authors: Rouse & Schieber
(M1, [000941](https://dandiarchive.org/dandiset/000941)); Nason-Tomaszewski,
Mender & Chestek (M2, [000953](https://dandiarchive.org/dandiset/000953));
Ye, Collinger & Gaunt (H1, [000954](https://dandiarchive.org/dandiset/000954));
and O'Doherty et al. (DANDI [000688](https://dandiarchive.org/dandiset/000688/0.250122.1735)).
The FALCON loader follows [`falcon-challenge`](https://github.com/snel-repo/falcon-challenge).

## DANDI688 experiments

Install the experiment dependencies (Python 3.10 or newer):

```bash
python -m pip install -e ".[experiments]"
python -m apst.data.download --tasks dandi688 --subjects sub-C sub-M
```

The experiments use **2015 center-out M1 sorted single units**, with separate
source training for each animal. They are within-animal cross-session
experiments; no Sub-C checkpoint is transferred to Sub-M.

| Protocol | Source sessions | Development sessions | Final sessions | Guide |
|----------|----------------:|---------------------:|---------------:|-------|
| Sub-C | 18 | 6 | 6 | [Sub-C reproduction](docs/dandi_subc.md) |
| Sub-M | 6 | 2 | 3 | [Sub-M reproduction](docs/dandi_subm.md) |

Each guide lists preparation, training, selection, final evaluation, and the
4/8/16/32-trial calibration-budget commands. APST uses association profiles at
both the E0 and token sites; ACT-only uses calibration activity without either
profile route. Target calibration does not update the APST or ACT model weights.
Output EMA has alpha 1/3 and runs only at inference on the full recording clock,
with its state reset for each session. Learnable recency slopes have zero weight
decay.

Source code and configurations are included; NWB data, prepared arrays,
checkpoints, and prediction files are not bundled. Download the public data and
train the models following the relevant guide. Historical numerical references
are recorded in [reference results](results/dandi688_reference.json); they are
not results of a new training run performed by installing this package.

`apst.models` provides the shared set-temporal decoder, recency implementation,
and pooled-carrier FiLM. `apst.legacy_dandi` contains the shared DANDI loading and
profile utilities; `apst.dandi_subc` and `apst.dandi_subm` preserve the two
experiment protocols separately.
