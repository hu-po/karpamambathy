FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    git \
    python3 \
    python3-pip \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && apt-get clean 
# install pip for python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip
# set python 3.11 as the default python and pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip $(which pip3.11) 1
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# karpathy-specific
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install \
    tiktoken==0.7.0 \
    datasets==2.20.0
# mamba-specific
RUN pip install \
    packaging==24.1 \
    causal-conv1d>=1.2.0
# extras (old protobuf for wandb)
RUN pip install \
    wandb==0.12.4 \
    protobuf==3.20.1 \
    hyperopt==0.2.7
RUN git clone https://github.com/state-spaces/mamba.git src/mamba
RUN pip install --no-cache-dir src/mamba
RUN mkdir /logs
RUN git config --global --add safe.directory /src
WORKDIR /src
COPY . /src
CMD ["python", "train.py"]