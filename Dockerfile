# COMPUTE Method 1: paste this whole file. Image name suggestion: urop-dsl-cu128
# Supports NVIDIA Blackwell (sm_120). Code stays in /project; do not COPY sources here.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir pytest matplotlib

# Sanity: image ships torch with CUDA; Blackwell smoke is verified at job time.
WORKDIR /workspace
