"""B1 (reviewer round 1): single-label results table, to support (or qualify) the
'on single-label tasks these mechanisms cluster' claim with actual numbers.
Reads experiments/full/results_{cifar,cifar100}_a{0.1,0.5}.json (committed) and emits
a compact CIFAR-10/100 table. Last-10-round mean, 3 seeds, mean+-pstd.
    python aggregate_sl.py
"""
import json, os, statistics

FULL = os.environ.get("FEDPRIOR_FULL", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "full"))
ROWS = [("FedAvg", "FedAvg"), ("FedProx", "FedProx"), ("SCAFFOLD", "SCAFFOLD"),
        ("FedSAM", "FedSAM"), ("FedNTD (masking)", "FedNTD"),
        ("FedPrior (blend)", "FedPrior")]
COLS = [("C10-.1", "cifar_a0.1"), ("C10-.5", "cifar_a0.5"),
        ("C100-.1", "cifar100_a0.1"), ("C100-.5", "cifar100_a0.5")]


# FedProx on the deep CIFAR tasks uses the FIXED-mu=0.1 re-tuned runs (the adaptive-mu
# runs in results_*.json ran mu away to ~29 and diverged); same override as aggregate.py.
RETUNE = json.load(open(os.path.join(FULL, "results_fedprox_retune.json")))


def ms(tag, key):
    if key == "FedProx" and isinstance(RETUNE.get(tag), list):
        fs = RETUNE[tag]                       # already per-seed last-10-round mean (%)
        return statistics.mean(fs), statistics.pstdev(fs)
    d = json.load(open(os.path.join(FULL, f"results_{tag}.json")))
    if key not in d:
        return None
    fs = [statistics.mean(h["test_acc"][-10:]) * 100 for h in d[key]]
    return statistics.mean(fs), (statistics.pstdev(fs) if len(fs) > 1 else 0.0)


lines = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Single-label accuracy (\%)}, GroupNorm ResNet-18, "
         r"Dirichlet label skew, 3 seeds. On single-label tasks the output-space "
         r"mechanisms cluster with FedAvg---\emph{except} that not-true masking "
         r"(FedNTD) \emph{helps} under extreme few-class skew (C10-.1), the regime "
         r"it was designed for (Sec.~\ref{sec:why}); on multi-label it instead "
         r"collapses. Balancing (BalBCE) is not applicable: softmax classes are "
         r"mutually exclusive, so there is no negative-weighting choice to make.}",
         r"\label{tab:sl}", r"\begin{tabular}{lcccc}", r"\toprule",
         "Method & " + " & ".join(c for c, _ in COLS) + r" \\", r"\midrule"]
for disp, key in ROWS:
    cells = []
    for _, tag in COLS:
        m = ms(tag, key)
        cells.append(f"{m[0]:.1f}\\tiny$\\pm${m[1]:.1f}" if m else "--")
    lines.append(disp + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = "\n".join(lines)
print(out)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                  "table_sl.tex"), "w").write(out + "\n")
print("\nAGG_SL_DONE")
