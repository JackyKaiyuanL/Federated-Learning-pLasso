"""M4 (reviewer round 1): the clean causal control for the margin figure.

The existing figure restores the negative term in BALANCED form (FedNTD-BCE), which
conflates 'restore negative supervision' with 'balance it'. Here we add the CLEAN
control: FedND-std = not-true masking with the negative BCE restored in STANDARD
(unbalanced) form (+ the same negative distillation). If FedND-std ALONE recovers the
margin/mAP, the collapse is caused by REMOVAL of negative supervision, independent of
balancing; balancing then supplies the extra mAP lift (FedND-std -> FedND).

Reuses the committed curves (FedAvg/FedNTD/FedNTD-BCE/FedPrior) and only runs the new
FedND-std curve. Writes results_diag_m4.json and a 5-curve figure into figures/.
    PYTHONPATH=. python -m experiments.diag_m4
"""
import json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "5")))
import fedprior
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert os.path.abspath(fedprior.__file__).startswith(_root), f"wrong fedprior: {fedprior.__file__}"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fedprior.algorithms import METHODS, federated
from experiments.diagnostic_ml import _build, _margin, _run, ROUNDS

HERE = os.path.dirname(os.path.abspath(__file__))
# Original 4-curve diagnostic (FedAvg/FedNTD/FedNTD-BCE/FedPrior); regenerate via
# diagnostic_ml.py, or point FEDPRIOR_DIAG at it. The committed FedND-std curve
# (results_diag_m4.json, shipped) is loaded first, so this is only a fallback.
ORIG = os.environ.get("FEDPRIOR_DIAG", os.path.join(HERE, "results_diagnostic_ml.json"))
FIG = os.environ.get("FEDPRIOR_FIG", os.path.join(HERE, "..", "figures"))   # artifact-local figures/
LOCAL = os.path.join(HERE, "..", "results_diag_m4.json")

out = json.load(open(LOCAL)) if os.path.exists(LOCAL) else json.load(open(ORIG))
if "FedND-std" not in out:
    cfg, cl, gc, tc = _build()
    out["FedND-std"] = _run(METHODS["FedND-std"], cfg, cl, gc, tc)
    json.dump(out, open(LOCAL, "w"))
print("[M4] FedND-std final mAP=%.1f margin=%.3f  | FedNTD %.1f/%.3f  FedND(bal) %.1f/%.3f"
      % (out["FedND-std"]["mAP"], out["FedND-std"]["margin"][-1],
         out["FedNTD"]["mAP"], out["FedNTD"]["margin"][-1],
         out["FedNTD-BCE"]["mAP"], out["FedNTD-BCE"]["margin"][-1]), flush=True)

plt.figure(figsize=(6.0, 4.0))
style = {"FedAvg": ("#1f77b4", "-"), "FedNTD": ("#ff7f0e", "-"),
         "FedND-std": ("#9467bd", "-."), "FedNTD-BCE": ("#2ca02c", "--"),
         "FedPrior": ("#d62728", "-")}
label = {"FedAvg": "FedAvg", "FedNTD": "FedNTD (mask, no neg BCE)",
         "FedND-std": "restore neg BCE, standard", "FedNTD-BCE": "restore neg BCE, balanced",
         "FedPrior": "FedPrior (blend)"}
for name, (c, ls) in style.items():
    d = out[name]
    plt.plot(d["rounds"], d["margin"], color=c, linestyle=ls, linewidth=2.0,
             label=f"{label[name]} (mAP {d['mAP']:.1f})")
plt.axhline(0, color="gray", lw=0.8, ls=":")
plt.xlabel("Communication round")
plt.ylabel("Positive$-$negative score margin")
plt.title("VOC Dir(0.1): masking collapses the margin; restoring the\n"
          "negative BCE---standard OR balanced---reverses it")
plt.legend(fontsize=7.5); plt.grid(alpha=0.3)
p1 = os.path.join(FIG, "diag_margin_voc.png")
plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
print("saved", p1, flush=True)
print("DIAG_M4_DONE", flush=True)
