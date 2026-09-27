# APST

Association profile conditioning for cross-session motor decoding with a
set-temporal transformer and no target-session gradient updates.

## Datasets

| Dataset | Public data |
| --- | --- |
| FALCON M1 | [DANDI 000941](https://dandiarchive.org/dandiset/000941) |
| FALCON M2 | [DANDI 000953](https://dandiarchive.org/dandiset/000953) |
| FALCON H1 | [DANDI 000954](https://dandiarchive.org/dandiset/000954) |
| DANDI688 SUA, Sub-C and Sub-M | [DANDI 000688](https://dandiarchive.org/dandiset/000688/0.250122.1735) |

## Getting started

Python 3.10 or newer:

```bash
python -m pip install -e ".[experiments]"
python -m apst.data.download --tasks dandi688 --subjects sub-C sub-M
```

Reproduction guides: [Sub-C](docs/dandi_subc.md),
[Sub-M](docs/dandi_subm.md), and [RNN baselines](docs/dandi_rnn.md).
They cover training, evaluation, and calibration-budget experiments.
The RNN baseline follows FALCON; its reference and DANDI settings are documented
in the RNN guide.

For FALCON data, run `python -m apst.data.download --tasks m1 m2 h1`.
The FALCON loader additionally requires
[`falcon-challenge`](https://github.com/snel-repo/falcon-challenge).

## Acknowledgements

We thank the original dataset authors and [DANDI](https://dandiarchive.org/)
for sharing the recordings, the [FALCON](https://snel-repo.github.io/falcon)
organizers for the benchmark and baseline implementations, and
[EvalAI](https://eval.ai/web/challenges/challenge-page/2319/overview)
for hosting the evaluation.

## Citation

If you use this code, please cite our APST paper. The arXiv link and BibTeX
entry will be added when the preprint is available.

## Contact

[zhangxy8@connect.hku.hk](mailto:zhangxy8@connect.hku.hk)
