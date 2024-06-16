import argparse
import os
import pprint
import subprocess

import numpy as np
import yaml
from hyperopt import fmin, hp, tpe

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch', type=int, default=4)

RESULTS_FILEPATH: str = "log/results.yaml"

# Define the search space
HYPERPARAMS = {
    # Model
    'hybrid_mode': hp.choice('hybrid_mode', [
        1,
        0
    ]),
    'data_aug': hp.choice('data_aug', [
        1,
        0
    ]),
    'mamba_d_state': hp.choice('mamba_d_state', [
        8,
        16,
        64
    ]),
    'att_n_embd': hp.choice('att_n_embd', [
        64,
        128
    ]),
    'n_layer': hp.choice('n_layer', [
        4,
        6,
        8
    ]),
    'warmup_frac': hp.choice('warmup_frac', [
        0.5,
        0.1
    ]),
    'max_lr': hp.choice('max_lr', [
        6e-4,
        1e-4,
        1e-3
    ]),
    'max_steps': hp.choice('max_steps', [
        256,
        512,
        1024
    ]),
    'weight_decay': hp.choice('weight_decay', [
        0.1,
        0.01,
        0.001
    ]),
    'grad_norm_clip': hp.choice('grad_norm_clip', [
        1.0,
        0.6,
        2.0
    ]),
}

def experiment(hparams) -> float:

    # Print hyperparam dict with logging
    print("\n\n Starting experiment \n\n")
    print(f"\n\nHyperparams:\n\n{pprint.pformat(hparams)}\n\n")


    if os.path.exists(RESULTS_FILEPATH):
        os.remove(RESULTS_FILEPATH)
    os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
    train_docker_proc = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "--gpus=all",
            "-v",
            f"{os.getcwd()}:/src",
            "-v",
            f"{os.getcwd()}/log:/log",
            "-e",
            f"WANDB_API_KEY={os.environ['WANDB_API_KEY']}",
            "karpamambathy",
            "python3",
            "train.py",
            f"--seed={args.seed}",
            f"--micro_batch_size={args.batch}",
            f"--hybrid_mode={hparams['hybrid_mode']}",
            f"--data_aug={hparams['data_aug']}",
            f"--mamba_d_state={hparams['mamba_d_state']}",
            f"--att_n_embd={hparams['att_n_embd']}",
            f"--n_layer={hparams['n_layer']}",
            f"--warmup_frac={hparams['warmup_frac']}",
            f"--max_lr={hparams['max_lr']}",
            # "--max_steps=2", # DEBUG
            f"--max_steps={hparams['max_steps']}",
            f"--weight_decay={hparams['weight_decay']}",
            f"--grad_norm_clip={hparams['grad_norm_clip']}",
        ]
    )
    train_docker_proc.wait()
    if train_docker_proc.returncode != 0:
        print("Error occurred when training")
        val_loss = 10.0
    else:
        print("Training completed, success")
        with open(RESULTS_FILEPATH, "r") as f:
            results = yaml.safe_load(f)
        val_loss = results["val_loss"]
    print(f"\n\nVal loss: {val_loss}\n\n")
    return val_loss

if __name__ == "__main__":
    args = parser.parse_args()
    HYPERPARAMS['seed'] = args.seed
    HYPERPARAMS['batch_size'] = args.batch
    best = fmin(
        experiment,
        space=HYPERPARAMS,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.Generator(np.random.PCG64(args.seed)),
    )