# APST

Code for **APST** (association profile conditioning) on the
[FALCON](https://snel-repo.github.io/falcon) few-shot neural decoding
benchmark and on [DANDI 000688](https://dandiarchive.org/dandiset/000688/0.250122.1735)
SUA reaching recordings. Training and decoder code will be added later. This
snapshot covers how to download and load the public datasets.

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
