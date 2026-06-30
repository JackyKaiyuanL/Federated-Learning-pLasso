"""GATE experiment for the FedND (FedNTD-BCE) paper: does the multi-label result
survive an ImageNet-PRETRAINED backbone (frozen BN), or is it a from-scratch
artifact? Runs the key methods on VOC + COCO (Dir 0.1 first, then 0.5), 3 seeds.

If FedNTD-BCE still clearly tops FedMLP/FedAvg here, the method paper is viable.

    python -m experiments.pretrained_gate
"""
import json, os, statistics
from fedprior.algorithms import TrainConfig, METHODS, federated
from fedprior import data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full")
os.makedirs(OUT, exist_ok=True)

# extreme-skew cells first (the gate), then the milder ones
CONFIGS = [("voc", 0.1), ("coco", 0.1), ("voc", 0.5), ("coco", 0.5)]
METHODS_RUN = ["FedAvg", "FedNTD", "FedMLP", "FedNTD-BCE", "FedPrior"]
SEEDS = (0, 1, 2)
ROUNDS = 60          # pretrained fine-tunes fast


def _cfg(task, seed):
    ncls = 80 if task == "coco" else 20
    return TrainConfig(model="resnet_ml_pt", model_kw={"num_classes": ncls},
                       lr=0.003, momentum=0.9, sam_rho=None, augment=True,
                       lr_schedule="cosine", batchsize=32, local_steps=5,
                       select_clients=8, device="cuda", seed=seed, eval_batch=128,
                       multilabel=True)


def _build(task, alpha, seed):
    if task == "voc":
        return data.build_voc_clients(40, alpha, seed)
    return data.build_coco_clients(40, alpha, seed)


def main():
    for task, alpha in CONFIGS:
        tag = f"pt_{task}_a{alpha}"
        path = os.path.join(OUT, f"results_{tag}.json")
        per = json.load(open(path)) if os.path.exists(path) else {}
        for name in METHODS_RUN:
            done = per.get(name, [])
            if len(done) >= len(SEEDS):
                continue
            for si, seed in enumerate(SEEDS):
                if si < len(done):
                    continue
                cfg = _cfg(task, seed)
                cl, gc, tc = _build(task, alpha, seed)
                _, h = federated(METHODS[name], ROUNDS, cl, gc, tc, cfg)
                done.append(h)
                per[name] = done
                json.dump(per, open(path, "w"))
                print(f"[{tag} s{seed}] {name:12s} mAP={h['test_acc'][-1]:.4f}", flush=True)
        m = {k: statistics.mean(statistics.mean(h['test_acc'][-10:])*100 for h in per[k])
             for k in per}
        print(f"[{tag}] means: " + "  ".join(f"{k}={v:.1f}" for k, v in m.items()), flush=True)
    print("PRETRAINED_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
