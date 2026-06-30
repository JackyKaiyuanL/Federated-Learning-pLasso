"""B2 (reviewer round 1): orthogonality check for the 'balancing is the lever' claim.
Add class-balanced BCE to the dedicated method FedMLP (FedMLP-bal) and re-run on the
PRETRAINED backbone (the realistic regime where the headline claim lives), VOC+COCO,
Dir(0.1) and Dir(0.5), 3 seeds. If FedMLP-bal climbs from FedMLP's ~14/61 to the
BalBCE/FedND level (~36/69), balancing is the lever AND is orthogonal/compatible with
FedMLP's teacher-consistency machinery -- a stronger, more honest story than a head-to-head.

Writes results to results_b2/ (shipped with this artifact).
    PYTHONPATH=. python -m experiments.b2_orthogonality
"""
import json, os, sys, statistics
# Make this artifact's own package win over any like-named package already on the path,
# while keeping cwd as the data root so data.py's relative "./data" still resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
import fedprior
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert os.path.abspath(fedprior.__file__).startswith(_root), f"wrong fedprior: {fedprior.__file__}"
from fedprior.algorithms import METHODS, federated
from experiments.pretrained_gate import _cfg, _build, ROUNDS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results_b2")
os.makedirs(OUT, exist_ok=True)
CONFIGS = [("voc", 0.1), ("voc", 0.5), ("coco", 0.1), ("coco", 0.5)]
SEEDS = (0, 1, 2)
NAME = "FedMLP-bal"

for task, alpha in CONFIGS:
    tag = f"pt_{task}_a{alpha}"
    path = os.path.join(OUT, f"results_{tag}.json")
    per = json.load(open(path)) if os.path.exists(path) else {}
    done = per.get(NAME, [])
    for si, seed in enumerate(SEEDS):
        if si < len(done):
            continue
        cfg = _cfg(task, seed)
        for attempt in range(4):                      # retry transient image-read errors
            try:
                cl, gc, tc = _build(task, alpha, seed); break
            except Exception as e:
                print(f"  [build retry {attempt}] {type(e).__name__}: {e}", flush=True)
        _, h = federated(METHODS[NAME], ROUNDS, cl, gc, tc, cfg)
        done.append(h)
        per[NAME] = done
        json.dump(per, open(path, "w"))
        print(f"[{tag} s{seed}] {NAME} mAP={h['test_acc'][-1]:.4f}", flush=True)
    m = statistics.mean(statistics.mean(h["test_acc"][-10:]) * 100 for h in per[NAME])
    print(f"[{tag}] {NAME} last10-mean mAP = {m:.2f}", flush=True)
print("B2_ORTHOGONALITY_DONE", flush=True)
