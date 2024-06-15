# karpamambathy

hybrid mamba transformer on arc challenge


## Setup - Local Conda

```bash
git clone git@github.com:hu-po/karpamambathy
conda create -y --name karpamambathy python=3.11
conda activate karpamambathy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken==0.7.0
pip install datasets==2.20.0
cd karpamambathy
git clone https://github.com/state-spaces/mamba.git
pip install packaging==24.1
pip install causal-conv1d>=1.2.0
cd mamba && pip install -e . && cd ..
pip install wandb==0.17.1
```

sanity check

```bash
./sanitycheck.sh
```

test dependencies

```bash
python3 train.py
```

## Setup - Docker

build the container

```bash
docker build -t karpamambathy -f Dockerfile .
```

sanity check

```bash
docker run -it --rm \
    --gpus device=0 \
    karpamambathy \
    ./sanitycheck.sh
```

ensure train works

```bash
docker run -it --rm \
    --gpus device=0 \
    -v $(pwd):/src \
    karpamambathy \
    python3 train.py
```

ensure wandb

```bash
docker run -it --rm \
    --gpus device=0 \
    -v $(pwd):/src \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    karpamambathy-practice \
    wandb login && python3 test_wandb.py
```