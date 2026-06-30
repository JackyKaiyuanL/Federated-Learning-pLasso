"""Comprehensive bake-off across all testbeds.

Methods: FedAvg, FedProx, SCAFFOLD, FedNTD, FedSAM, and the two FedPrior
FedPrior (pLasso blend+L1 + per-class + SAM) and the not-true variant FedPrior-NT
(not-true masking + per-class + SAM).

Tasks: synthetic, mnist, cifar, voc (single-label use accuracy; voc uses mAP).
Results saved per (task, alpha) to experiments/full/results_<tag>.json as
{method: [history]} so postprocessing/aggregation is uniform.

    python -m experiments.full_benchmark --tasks cifar --alphas 0.1 0.5
    python -m experiments.full_benchmark --tasks synthetic mnist cifar
"""
import argparse
import json
import os

import torch
from fedprior.algorithms import TrainConfig, METHODS, federated, Global

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full")
CKPT = os.path.join(OUT, "ckpt")
os.makedirs(CKPT, exist_ok=True)

BAKEOFF = ["FedAvg", "FedProx", "SCAFFOLD", "FedNTD", "FedSAM", "FedPrior"]
ML_ONLY = ["FedAwS", "FedMLP"]      # dedicated multi-label FL baselines (VOC)


def _cfg(**kw):
    return TrainConfig(**kw)


def build_task(task, alpha, seed):
    """Return (train_cfg, clients, global_client, test_client, rounds, with_global)."""
    from fedprior import data
    if task == "synthetic":
        cfg = _cfg(model="linear", lr=5e-4, batchsize=10, local_steps=20,
                   select_clients=10, straggler_rate=0.5, device="cuda", seed=seed)
        X, y, n = data.generate_synthetic(1.0, 1.0, iid=False, num_users=30, seed=seed)
        cl, gc, tc = data.build_clients(X, y, n, 30)
        return cfg, cl, gc, tc, 200, True
    if task == "mnist":
        cfg = _cfg(model="cnn", lr=0.01, batchsize=64, local_steps=5,
                   select_clients=10, device="cuda", seed=seed)
        cl, gc, tc = data.build_mnist_clients(100, alpha, seed)
        return cfg, cl, gc, tc, 100, True
    if task == "cifar":
        cfg = _cfg(model="resnet", model_kw={"num_classes": 10}, lr=0.03,
                   momentum=0.9, sam_rho=0.02, augment=True, lr_schedule="cosine",
                   batchsize=64, local_steps=5, select_clients=10, device="cuda",
                   seed=seed, eval_batch=1000)
        cl, gc, tc = data.build_cifar_clients(100, alpha, seed)
        return cfg, cl, gc, tc, 600, False
    if task == "cifar100":
        cfg = _cfg(model="resnet", model_kw={"num_classes": 100}, lr=0.03,
                   momentum=0.9, sam_rho=0.02, augment=True, lr_schedule="cosine",
                   batchsize=64, local_steps=5, select_clients=10, device="cuda",
                   seed=seed, eval_batch=1000)
        cl, gc, tc = data.build_cifar100_clients(100, alpha, seed)
        return cfg, cl, gc, tc, 600, False
    if task == "coco":
        cfg = _cfg(model="resnet_ml", model_kw={"num_classes": 80}, lr=0.01,
                   momentum=0.9, sam_rho=0.02, augment=True, lr_schedule="cosine",
                   batchsize=32, local_steps=5, select_clients=8, device="cuda",
                   seed=seed, eval_batch=128, multilabel=True)
        cl, gc, tc = data.build_coco_clients(40, alpha, seed)
        return cfg, cl, gc, tc, 150, False
    if task == "voc":
        cfg = _cfg(model="resnet_ml", lr=0.01, momentum=0.9, sam_rho=0.02,
                   augment=True, lr_schedule="cosine", batchsize=32, local_steps=5,
                   select_clients=8, device="cuda", seed=seed, eval_batch=128,
                   multilabel=True)
        cl, gc, tc = data.build_voc_clients(40, alpha, seed)
        return cfg, cl, gc, tc, 100, False
    raise ValueError(task)


def run_task(task, alpha, seeds=(0, 1, 2)):
    """Run every method over `seeds`, appending one history per seed. Reuses
    histories already saved (index = seed position), so adding seeds is cheap."""
    tag = (f"{task}_a{alpha}" if task in ("mnist","cifar","cifar100","voc","coco") else task)
    path = os.path.join(OUT, f"results_{tag}.json")
    per = json.load(open(path)) if os.path.exists(path) else {}

    for si, seed in enumerate(seeds):
        cfg, cl, gc, tc, rounds, with_global = build_task(task, alpha, seed)
        names = (["Global"] if with_global else []) + BAKEOFF
        if task in ("voc", "coco"):
            names = names + ML_ONLY
        for name in names:
            if len(per.get(name, [])) > si:          # this seed already done
                continue
            if name == "Global":
                state, h = Global(rounds, gc, tc, cfg)
            else:
                state, h = federated(METHODS[name], rounds, cl, gc, tc, cfg)
            per.setdefault(name, []).append(h)
            json.dump(per, open(path, "w"))
            # save final weights -> future runs can warm-restart (true resume)
            torch.save({k: v.cpu() for k, v in state.items()},
                       os.path.join(CKPT, f"{tag}_{name}_s{seed}.pt"))
            metric = "mAP" if cfg.multilabel else "acc"
            print(f"[{tag} s{seed}] {name:12s} {metric}={h['test_acc'][-1]:.4f}",
                  flush=True)
    print(f"[{tag}] DONE ({len(seeds)} seeds)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["synthetic", "mnist", "cifar", "voc"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    for task in args.tasks:
        if task in ("mnist","cifar","cifar100","voc","coco"):
            for a in args.alphas:
                run_task(task, a, tuple(args.seeds))
        else:
            run_task(task, None, tuple(args.seeds))
    print("FULL_BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
