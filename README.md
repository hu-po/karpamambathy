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
pip install hyperopt==0.2.7
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
    karpamambathy \
    python3 test_wandb.py
```

# Sweeping

```bash
python3 sweep.py --seed=42
```

## Ideas

- [x] make public repo
- [x] copy over environment setup
- [x] sanity check
- [x] replace GPT Blocks with Mamba2 Blocks (optimal amount of mamba blocks based on paper)
- [x] change micro batch size for gpu
- [x] hybrid model with transformer and mamba2 blocks?
- [x] test locally and confirm working
- [x] dataloader with https://github.com/fchollet/ARC-AGI
- [x] shuffling dataloader
- [x] bugfix for mamba on github issues
- [x] wandb for logging
- [x] wandb plotting and sweeping over hyperparams
- [x] Gpt-4o versus gpt-4 on loss curves
- [x] check hyperparams with papers
- [x] Looking through paper for sweep ideas
- [x] Reversed, flipped examples as data augmentation
- [x] Pad as separate token
- [ ] sshed into other computer, running on docker
- [ ] timing and gpu usage
- [ ] Weave in distillation from pretrained phi-3 mamba

## References

https://arxiv.org/pdf/2406.07887
https://arxiv.org/pdf/2405.21060
https://arxiv.org/pdf/1911.01547

https://github.com/microsoft/Samba/
https://github.com/karpathy/build-nanogpt
https://github.com/state-spaces/mamba
https://github.com/fchollet/ARC-AGI