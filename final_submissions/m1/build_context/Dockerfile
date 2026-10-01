ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2
FROM ${BASE_IMAGE}
COPY payload /payload
COPY decode.py /decode.py
COPY entrypoint.sh /entrypoint.sh
COPY fixture_validate.py /fixture_validate.py
ENV PYTHONPATH=/payload OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--evaluation", "remote", "--model-path", "/payload", "--split", "m1", "--phase", "test", "--batch-size", "4"]
