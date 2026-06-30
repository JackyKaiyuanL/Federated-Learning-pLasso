"""FedPrior: a prior-LASSO (pLasso) style federated learning method --
a per-class-adaptive, response-space prior (distillation toward the global
model) optimized with a sharpness-aware local step -- plus baselines
(FedAvg, FedProx, SCAFFOLD, FedNTD, FedSAM).
"""

from .models import Client, LinearNet, SmallCNN, build_model
from .data import (
    generate_synthetic, build_clients,
    load_mnist, dirichlet_partition, build_mnist_clients,
)
from .algorithms import (
    TrainConfig, MethodSpec, METHODS, LocalTrainer, evaluate, global_loss,
    federated, run_method, Global,
    FedAvg, FedProx, SCAFFOLD, FedNTD, FedSAM, FedPrior,
)
from .run import ExperimentConfig, run_experiment, plot_history, BENCH_METHODS

__all__ = [
    "Client", "LinearNet", "SmallCNN", "build_model",
    "generate_synthetic", "build_clients",
    "load_mnist", "dirichlet_partition", "build_mnist_clients",
    "TrainConfig", "MethodSpec", "METHODS", "LocalTrainer",
    "evaluate", "global_loss", "federated", "run_method", "Global",
    "FedAvg", "FedProx", "SCAFFOLD", "FedNTD", "FedSAM", "FedPrior",
    "ExperimentConfig", "run_experiment", "plot_history", "BENCH_METHODS",
]
