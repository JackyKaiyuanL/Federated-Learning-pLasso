"""Experiment orchestration: run methods on the synthetic or MNIST task.

Examples
--------
    # synthetic softmax benchmark
    python -m fedprior.run --task synthetic --rounds 200 --straggler 0.5 --device cuda

    # MNIST Dirichlet non-IID benchmark
    python -m fedprior.run --task mnist --dirichlet-alpha 0.1 --rounds 100 \
        --clients 100 --select 10 --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from .algorithms import TrainConfig, METHODS, federated, Global
from .data import generate_synthetic, build_clients, build_mnist_clients


# Methods reported in the benchmark (Global is added separately).
BENCH_METHODS = ["FedAvg", "FedProx", "SCAFFOLD", "FedNTD", "FedSAM", "FedPrior"]


@dataclass
class ExperimentConfig:
    task: str = "synthetic"          # 'synthetic' | 'mnist'
    # synthetic
    alpha: float = 1.0
    beta: float = 1.0
    iid: bool = False
    num_users: int = 30
    # mnist
    dirichlet_alpha: float = 0.5
    data_root: str = "./data"
    # shared
    rounds: int = 200
    seed: int = 0
    methods: tuple = tuple(BENCH_METHODS)
    train: TrainConfig = None  # type: ignore

    def __post_init__(self):
        if self.train is None:
            self.train = TrainConfig()


def _build_data(cfg: ExperimentConfig):
    if cfg.task == "synthetic":
        cfg.train.model = "linear"
        X, y, n = generate_synthetic(cfg.alpha, cfg.beta, iid=cfg.iid,
                                     num_users=cfg.num_users, seed=cfg.seed)
        return build_clients(X, y, n, cfg.num_users)
    elif cfg.task == "mnist":
        cfg.train.model = "cnn"
        return build_mnist_clients(num_clients=cfg.num_users,
                                   alpha=cfg.dirichlet_alpha, seed=cfg.seed,
                                   root=cfg.data_root)
    raise ValueError(f"unknown task '{cfg.task}'")


def run_experiment(cfg: ExperimentConfig, include_global=True) -> dict:
    """Run the requested methods on one config. Returns {method: (state, history)}."""
    clients, global_client, test_client = _build_data(cfg)
    results = {}
    names = (["Global"] if include_global else []) + list(cfg.methods)
    for name in names:
        if name == "Global":
            results[name] = Global(cfg.rounds, global_client, test_client, cfg.train)
        else:
            results[name] = federated(METHODS[name], cfg.rounds, clients,
                                      global_client, test_client, cfg.train)
        h = results[name][1]
        print(f"{name:11s} | test_acc={h['test_acc'][-1]:.4f} "
              f"train_acc={h['train_acc'][-1]:.4f} loss={h['loss'][-1]:.4f}",
              flush=True)
    return results


def plot_history(results: dict, metric="test_acc", title="", save=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 6))
    for name, (_, hist) in results.items():
        plt.plot(hist[metric], label=name, linewidth=2)
    plt.xlabel("Communication round")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    if save:
        plt.savefig(save, dpi=130, bbox_inches="tight")
    return plt


def _argparser():
    p = argparse.ArgumentParser(description="FedPrior experiments")
    p.add_argument("--task", choices=["synthetic", "mnist"], default="synthetic")
    p.add_argument("--alpha", type=float, default=1.0, help="synthetic alpha")
    p.add_argument("--beta", type=float, default=1.0, help="synthetic beta")
    p.add_argument("--iid", action="store_true")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5,
                   dest="dir_alpha", help="MNIST Dirichlet alpha")
    p.add_argument("--clients", type=int, default=30, dest="num_users")
    p.add_argument("--select", type=int, default=10)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--local-steps", type=int, default=20)
    p.add_argument("--batchsize", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--straggler", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    a = _argparser().parse_args(argv)
    train = TrainConfig(lr=a.lr, batchsize=a.batchsize, local_steps=a.local_steps,
                        select_clients=a.select, straggler_rate=a.straggler,
                        device=a.device, seed=a.seed)
    cfg = ExperimentConfig(task=a.task, alpha=a.alpha, beta=a.beta, iid=a.iid,
                           dirichlet_alpha=a.dir_alpha, num_users=a.num_users,
                           rounds=a.rounds, seed=a.seed, data_root=a.data_root,
                           train=train)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
