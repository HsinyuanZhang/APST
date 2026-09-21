# APST

Code for **APST** (association profile conditioning) on the
[FALCON](https://snel-repo.github.io/falcon) few-shot neural decoding
benchmark. Training and decoder code will be added here. This first snapshot
covers how to get and load the public FALCON recordings.

## Data

APST uses the three FALCON motor tasks hosted on [DANDI](https://dandiarchive.org/).
These are the same dandisets as the official [EvalAI FALCON challenge](https://eval.ai/web/challenges/challenge-page/2319/overview).

| Task | DANDI | Subject folder |
|------|-------|----------------|
| M1 | [000941](https://dandiarchive.org/dandiset/000941) | `sub-MonkeyL` |
| M2 | [000953](https://dandiarchive.org/dandiset/000953) | `sub-MonkeyN` |
| H1 | [000954](https://dandiarchive.org/dandiset/000954) | `sub-HumanPitt` |

Each dandiset has three public splits: `held-in-calib` (training),
`held-in-minival` (local sanity check), and `held-out-calib` (development
held-out). Official test labels stay on EvalAI and are not in these files.

### Install

```bash
python -m pip install -e .
python -m pip install "falcon-challenge @ git+https://github.com/snel-repo/falcon-challenge.git"
```

`dandi` is enough to download. `falcon-challenge` is required to load NWB files
with the official evaluator reader.

### Download

```bash
# all three tasks into ./data/<dandiset id>/
python -m apst.data.download

# one task, custom directory
python -m apst.data.download --tasks m1 --root /path/to/data
```

Set `APST_DATA_ROOT` (or `EVAL_DATA_PATH`, the FALCON evaluator variable) if
the files are not under `./data`.

Equivalent DANDI CLI:

```bash
dandi download https://dandiarchive.org/dandiset/000941 -o data
dandi download https://dandiarchive.org/dandiset/000953 -o data
dandi download https://dandiarchive.org/dandiset/000954 -o data
```

### Load

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

## Acknowledgements

We thank the FALCON organizers for the benchmark, the public NWB layout, and
the evaluation protocol, and the EvalAI team for hosting the
[FALCON challenge](https://eval.ai/web/challenges/challenge-page/2319/overview).
Recordings are released on DANDI by the original authors: Rouse & Schieber
(M1, [000941](https://dandiarchive.org/dandiset/000941)); Nason-Tomaszewski,
Mender & Chestek (M2, [000953](https://dandiarchive.org/dandiset/000953));
Ye, Collinger & Gaunt (H1, [000954](https://dandiarchive.org/dandiset/000954)).
The loader follows [`falcon-challenge`](https://github.com/snel-repo/falcon-challenge).
