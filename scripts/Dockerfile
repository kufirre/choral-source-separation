FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.org/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      gnupg \
      libsndfile1 \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/choral-source-separation

COPY requirements.txt requirements-vertex.txt ./

# Pin torch + CUDA deps to prevent pip from upgrading them when resolving
# transitive dependencies in requirements-vertex.txt.
RUN printf '%s\n' \
      'torch==2.2.2' \
      'torchvision==0.17.2' \
      'torchaudio==2.2.2' \
    > /tmp/torch-constraints.txt

RUN REQ_FILE=requirements-vertex.txt \
    && if [ ! -f "${REQ_FILE}" ]; then REQ_FILE=requirements.txt; fi \
    && python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        torch==2.2.2 \
        torchvision==0.17.2 \
        torchaudio==2.2.2 \
    && PIP_CONSTRAINT=/tmp/torch-constraints.txt \
       python -m pip install --no-build-isolation -r "${REQ_FILE}" \
    && python -m pip install --no-build-isolation causal-conv1d==1.4.0 \
    && python -m pip install --no-build-isolation --no-deps mamba-ssm==2.2.2 \
    && python -m pip install --no-build-isolation --no-deps "torch_log_wmse>=0.3.1" \
    && python -m pip install --no-build-isolation --no-deps hyper_connections==0.1.11

RUN python -c "import soundfile; import torch; print('torch', torch.__version__)"

COPY . /workspace/choral-source-separation

RUN chmod +x scripts/vertex/*.sh

ENTRYPOINT ["/workspace/choral-source-separation/scripts/vertex/entrypoint.sh"]
