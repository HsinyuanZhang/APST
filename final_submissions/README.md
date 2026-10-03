# Final FALCON submission checkpoints and Docker builds

Exact model artifacts for APST's final public submissions under team **sustechhku**.

| Task | EvalAI submission | Official held-out R² mean | Artifacts |
| --- | --- | --- | --- | --- | --- |
| M1 | 583237 | 0.6541798272722487 | [M1](m1/) |
| M2 | 583092 | 0.42337295173829836 | [M2](m2/) |
| H1 | 583097 | 0.4404963152029235 | [H1](h1/) |

[Official leaderboard](https://eval.ai/web/challenges/challenge-page/2319/leaderboard) · [Paper](https://arxiv.org/abs/2609.39080)

## Checkpoints and inference payloads

Each task directory includes the selected training checkpoint at
`checkpoints/source_checkpoint.pt`, the exact inference EMA decoder at
`build_context/payload/selected_ema_decoder.pt`, and the frozen association-profile
banks, configuration, and runtime code. The source checkpoint includes training
state; the packaged EMA decoder contains the inference weights. Their separate
SHA-256 hashes and the original image ID are recorded in `submission.json`.
The banks preserve the offline conditioning used in the submitted decoder.

From the repository root, verify every published file and the original payload
manifest bindings:

```bash
python3 final_submissions/verify.py
```

## Build Docker images

The task Dockerfiles use a public Python base image and pinned CPU runtime
packages. They do not require access to the original private submission registry.
From the repository root:

```bash
docker build -f final_submissions/m1/Dockerfile -t apst-m1:public-rebuild final_submissions/m1/build_context
docker build -f final_submissions/m2/Dockerfile -t apst-m2:public-rebuild final_submissions/m2/build_context
docker build -f final_submissions/h1/Dockerfile -t apst-h1:public-rebuild final_submissions/h1/build_context
```

See each task README for its runtime dependencies and entrypoint. The image
contains the packaged EMA decoder and frozen banks; training state is kept in
`checkpoints/` and is not copied into the inference image.

Each `build_context/Dockerfile` is the original build record. The public build
recipe creates a new CPU image with the submitted payload. Its image ID and
latency may differ from the original submission; `submission.json` records the
original image ID for provenance. This release distributes checkpoints and build
methods, without uploading image archives.

## Validation

All source checkpoints, packaged EMA decoders, and payload manifest file hashes
match the final submission records. Local validation fixtures and Python
bytecode are excluded from the source folders. Container validation compares the
rebuilt runtime against the original submission's streaming inference outputs;
see [validation.json](validation.json) for results and the scope of these checks.
