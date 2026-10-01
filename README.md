# APST

[![arXiv](https://img.shields.io/badge/arXiv-2609.39080-b31b1b.svg)](https://arxiv.org/abs/2609.39080)
[![FALCON Leaderboard](https://img.shields.io/badge/EvalAI-FALCON%20Leaderboard-0057B8.svg)](https://eval.ai/web/challenges/challenge-page/2319/leaderboard)

Association profile conditioning for cross-session motor decoding with a
set-temporal transformer and no target-session gradient updates.

**Paper:** [arXiv:2609.39080](https://arxiv.org/abs/2609.39080) · [PDF](https://arxiv.org/pdf/2609.39080)

**Official FALCON leaderboard:** [View public submissions and benchmark scores on EvalAI](https://eval.ai/web/challenges/challenge-page/2319/leaderboard).

**Final submission artifacts:** [M1, M2, and H1 checkpoints, frozen banks, runtime code, and Docker build instructions](final_submissions/).

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

If you use this code, please cite [our paper](https://arxiv.org/abs/2609.39080).

<details>
<summary>BibTeX</summary>

```bibtex
@misc{zhang2026association,
  title = {Association profile conditioning in a set-temporal transformer for cross-session intracortical motor decoding},
  author = {Zhang, Xinyuan and Mo, Handong and Wen, Pengfei and Liang, Shuang and Yang, Jichang and Zeng, Yan and Wang, Zhongrui and Wang, Han},
  year = {2026},
  eprint = {2609.39080},
  archivePrefix = {arXiv},
  primaryClass = {q-bio.NC},
  url = {https://arxiv.org/abs/2609.39080}
}
```

</details>

## Contact

[zhangxy8@connect.hku.hk](mailto:zhangxy8@connect.hku.hk)
