"""Mechanism diagnostic for the multi-label collapse of not-true distillation.

We track, per communication round on the VOC test set, the *score margin*
    margin = mean sigmoid score on POSITIVE (present) labels
           - mean sigmoid score on NEGATIVE (absent)  labels
averaged over classes. mAP is a ranking metric, so a healthy method must keep
this margin positive and growing; if the not-true mechanism is what kills mAP,
its margin must collapse toward zero while the pLasso blend's keeps separating.
If the curves match that prediction, the collapse is the mechanism, not an
artifact (an eval/data bug would hit FedAvg/FedPrior too, but they stay healthy).

Also runs an eta (distillation-strength) sweep for FedNTD to rule out
"the distillation was simply too strong" as the cause.

    python -m experiments.diagnostic_ml
"""
import json
import os
import statistics
from dataclasses import replace

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fedprior.algorithms import TrainConfig, MethodSpec, METHODS, federated
from fedprior import data

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(PAPER_FIG, exist_ok=True)

ROUNDS = 100
SEED = 0
EVERY = 2          # log the margin every EVERY rounds (keep it cheap)


def _build():
    cfg = TrainConfig(model="resnet_ml", lr=0.01, momentum=0.9, sam_rho=0.02,
                      augment=True, lr_schedule="cosine", batchsize=32,
                      local_steps=5, select_clients=8, device="cuda", seed=SEED,
                      eval_batch=128, multilabel=True)
    cl, gc, tc = data.build_voc_clients(40, 0.1, SEED)
    return cfg, cl, gc, tc


@torch.no_grad()
def _margin(state, cfg, test_client):
    """mean positive-label score minus mean negative-label score (over classes)."""
    from fedprior.models import build_model
    dev = torch.device(cfg.device)
    net = build_model(cfg.model, **cfg.model_kw).to(dev); net.load_state_dict(state); net.eval()
    n = test_client.number
    P, Y = [], []
    for i in range(0, n, cfg.eval_batch):
        xb = test_client.x[i:i + cfg.eval_batch].to(dev).float()
        P.append(torch.sigmoid(net(xb)).cpu())
        Y.append(test_client.y[i:i + cfg.eval_batch].cpu())
    P = torch.nan_to_num(torch.cat(P), nan=0.0); Y = torch.cat(Y)
    pos = P[Y == 1]; neg = P[Y == 0]
    pm = pos.mean().item() if pos.numel() else 0.0
    nm = neg.mean().item() if neg.numel() else 0.0
    return pm, nm


def _run(spec, cfg, cl, gc, tc):
    rounds_logged, pos_s, neg_s = [], [], []

    def hook(r, state):
        if r % EVERY == 0:
            pm, nm = _margin(state, cfg, tc)
            rounds_logged.append(r); pos_s.append(pm); neg_s.append(nm)

    _, h = federated(spec, ROUNDS, cl, gc, tc, cfg, on_round=hook)
    final_map = statistics.mean(h["test_acc"][-10:]) * 100
    return dict(rounds=rounds_logged, pos=pos_s, neg=neg_s,
                margin=[p - n for p, n in zip(pos_s, neg_s)], mAP=final_map)


def main():
    cfg, cl, gc, tc = _build()
    out = {}

    # 1) margin trajectories: FedAvg (healthy baseline), FedNTD (collapse),
    #    FedNTD-BCE (collapse REVERSED by restoring negative supervision), FedPrior (blend)
    for name in ["FedAvg", "FedNTD", "FedNTD-BCE", "FedPrior"]:
        out[name] = _run(METHODS[name], cfg, cl, gc, tc)
        print(f"[traj] {name:10s} final mAP={out[name]['mAP']:.1f} "
              f"final margin={out[name]['margin'][-1]:.3f}", flush=True)
        json.dump(out, open(os.path.join(HERE, "results_diagnostic_ml.json"), "w"))

    # 2) eta sweep for FedNTD (distillation strength): does a weaker term avoid collapse?
    base = METHODS["FedNTD"]
    for eta in [0.1, 0.3, 1.0]:
        spec = replace(base, name=f"FedNTD-eta{eta}", eta=eta)
        out[spec.name] = _run(spec, cfg, cl, gc, tc)
        print(f"[sweep] eta={eta:<4} final mAP={out[spec.name]['mAP']:.1f} "
              f"final margin={out[spec.name]['margin'][-1]:.3f}", flush=True)
        json.dump(out, open(os.path.join(HERE, "results_diagnostic_ml.json"), "w"))

    # --- plot: margin vs round (the money figure) ---
    plt.figure(figsize=(6.0, 4.0))
    style = {"FedAvg": ("#1f77b4", "-"), "FedNTD": ("#ff7f0e", "-"),
             "FedNTD-BCE": ("#2ca02c", "--"), "FedPrior": ("#d62728", "-")}
    for name, (c, ls) in style.items():
        d = out[name]
        plt.plot(d["rounds"], d["margin"], color=c, linestyle=ls, linewidth=2.0,
                 label=f"{name} (mAP {d['mAP']:.1f})")
    plt.axhline(0, color="gray", lw=0.8, ls=":")
    plt.xlabel("Communication round")
    plt.ylabel("Positive$-$negative score margin")
    plt.title("VOC Dir(0.1): not-true masking collapses the margin;\n"
              "restoring negative BCE (FedNTD-BCE) reverses it")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    p1 = os.path.join(PAPER_FIG, "diag_margin_voc.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print("saved", p1)

    print("DIAGNOSTIC_ML_DONE", flush=True)


if __name__ == "__main__":
    main()
