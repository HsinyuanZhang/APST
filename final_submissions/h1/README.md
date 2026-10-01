# H1 release material

This directory packages submission `583097` (`h1_ladder_x2_s44_ring_public_v1`). `build_context/` was copied from the original submission build context, with `fixture.npz`, `fixture_validate.py`, Python bytecode, and `__pycache__` excluded because they are not public release material. The original `build_context/Dockerfile` is retained only as a build record and can refer to excluded fixture files and its former private base image.

`checkpoints/source_checkpoint.pt` is the selected training checkpoint. `build_context/payload/selected_ema_decoder.pt` is the packaged decoder artifact; it is distinct from the source checkpoint. The payload's banks are already baked into the payload.

From the repository root, build the public runtime with `docker build -f final_submissions/h1/Dockerfile -t apst-h1:public-rebuild final_submissions/h1/build_context`. This Dockerfile uses `python:3.10-slim-bookworm`, installs CPU-only `torch==2.5.1+cpu` from the PyTorch CPU index, and installs the pinned runtime dependencies in `build_context/requirements-runtime.txt`. The resulting image is a rebuild and does not have the original submission image ID. Verify the release from the repository root with `python3 final_submissions/verify.py`.

For a standard remote evaluation, first create the output directory, then use the image default command:

```sh
mkdir -p /path/to/output
docker run --rm --network none \
  -v /path/to/evaluation_data:/dataset/evaluation_data:ro \
  -v /path/to/output:/submission \
  apst-h1:public-rebuild
```

The image sets the EvalAI data and prediction paths to these two mount points.
