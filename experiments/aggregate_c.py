"""Builds the paper tables (writes tables.tex).
Five tables: (1) pretrained ML mAP incl. FLAG, (2) from-scratch ML mAP incl. FLAG,
(3) 2x2 dissection, (4) single-label CIFAR, (5) orthogonality: +balanced BCE into
FedMLP/FedNTD. Reads the bake-off results experiments/full/*.json + local results_b2/*.json.
    PYTHONPATH=. python -m experiments.aggregate_c
"""
import json, os, statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # artifact root (experiments/, fedprior/, results_b2/)
# Shared bake-off results (experiments/full/*.json) are produced by full_benchmark.py;
# set FEDPRIOR_FULL to point at them, or place them under <root>/experiments/full.
FULL = os.environ.get("FEDPRIOR_FULL", os.path.join(_ROOT, "experiments", "full"))
B2 = os.path.join(_ROOT, "results_b2")               # shipped with this artifact
OUT = os.environ.get("FEDPRIOR_TABLES_OUT", os.path.join(_ROOT, "tables.tex"))
RETUNE = json.load(open(os.path.join(FULL, "results_fedprox_retune.json")))


def _hist_ms(path, key):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if key not in d:
        return None
    fs = [statistics.mean(h["test_acc"][-10:]) * 100 for h in d[key]]
    return statistics.mean(fs), (statistics.pstdev(fs) if len(fs) > 1 else 0.0), len(fs)


def ml(tag, key, pref=""):
    return _hist_ms(os.path.join(FULL, f"results_{pref}{tag}.json"), key)


def b2(tag, key):
    return _hist_ms(os.path.join(B2, f"results_{tag}.json"), key)


def cell(ms, n_flag=None):
    if ms is None:
        return "--"
    m, sd, n = ms
    s = f"{m:.1f}\\tiny$\\pm${sd:.1f}"
    return s + (f"\\tiny$^{{n{n}}}$" if n_flag and n < 3 else "")


COLS = [("VOC-.1", "voc_a0.1"), ("VOC-.5", "voc_a0.5"),
        ("COCO-.1", "coco_a0.1"), ("COCO-.5", "coco_a0.5")]


def ml_table(pref, caption, label, rows):
    best = {}
    for cn, tag in COLS:
        vals = {d: ml(tag, k, pref)[0] for d, k in rows if ml(tag, k, pref)}
        best[cn] = max(vals, key=vals.get) if vals else None
    L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         "Method & " + " & ".join(c for c, _ in COLS) + r" \\", r"\midrule"]
    for disp, key in rows:
        cs = []
        for cn, tag in COLS:
            ms = ml(tag, key, pref)
            txt = cell(ms, n_flag=(key == "FLAG"))
            if ms and disp == best[cn]:
                txt = r"\textbf{" + txt + "}"
            cs.append(txt)
        L.append(disp + " & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


ML_ROWS = [("FedAvg", "FedAvg"), ("FedNTD (masking)", "FedNTD"), ("FedAwS", "FedAwS"),
           ("FedMLP", "FedMLP"), ("FLAG", "FLAG"), ("FedPrior (blend)", "FedPrior"),
           ("\\textbf{BalBCE}", "BalBCE"), ("FedND (bal+distill)", "FedNTD-BCE")]


def twobytwo():
    L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\caption{Disentangling the multi-label gain: class-balanced BCE vs.\ "
         r"negative distillation (mAP, mean over 3 seeds, Dir$(0.1)$). Balancing is "
         r"the robust lever; negative distillation is situational (helps only "
         r"from-scratch VOC).}", r"\label{tab:2x2}",
         r"\begin{tabular}{llcc}", r"\toprule",
         r"Backbone & BCE & no distill & +neg distill \\", r"\midrule"]
    pairs = [("From-scratch", "", [("std", "FedAvg", "FedND-std"),
                                   ("balanced", "BalBCE", "FedNTD-BCE")]),
             ("Pretrained", "pt_", [("std", "FedAvg", "FedND-std"),
                                    ("balanced", "BalBCE", "FedNTD-BCE")])]
    for bk, pref, rows in pairs:
        for i, (bce, k_no, k_di) in enumerate(rows):
            def c(key):
                a = ml("voc_a0.1", key, pref); b = ml("coco_a0.1", key, pref)
                return f"{a[0]:.1f}/{b[0]:.1f}" if a and b else "--"
            L.append(f"{bk if i == 0 else ''} & {bce} & {c(k_no)} & {c(k_di)} \\\\")
        L.append(r"\midrule" if bk == "From-scratch" else "")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[2pt]{\footnotesize Cells are VOC/COCO mAP\%.}", r"\end{table}"]
    return "\n".join(x for x in L if x)


def ortho():
    # pretrained: FedMLP / FedMLP+bal / FedNTD / FedNTD+bal(=FedND) / BalBCE
    rows = [("FedAvg (std BCE)", lambda t: ml(t, "FedAvg", "pt_")),
            ("FedMLP", lambda t: ml(t, "FedMLP", "pt_")),
            ("\\quad +\\,balanced BCE", lambda t: b2(f"pt_{t}", "FedMLP-bal")),
            ("FedNTD (masking)", lambda t: ml(t, "FedNTD", "pt_")),
            ("\\quad +\\,balanced BCE", lambda t: ml(t, "FedNTD-BCE", "pt_")),
            ("\\textbf{BalBCE}", lambda t: ml(t, "BalBCE", "pt_"))]
    L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Orthogonality check} (pretrained backbone, mAP \%, 3 "
         r"seeds): adding class-balanced BCE \emph{into} the dedicated methods. FedMLP "
         r"and FedNTD both omit balancing; adding it lifts each toward the BalBCE/FedND "
         r"band, showing balancing is an orthogonal, compatible lever rather than a "
         r"competitor. (FedNTD\,+\,balanced is exactly our FedND.)}", r"\label{tab:ortho}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         "Method & " + " & ".join(c for c, _ in COLS) + r" \\", r"\midrule"]
    for disp, fn in rows:
        cs = [cell(fn(tag)) for _, tag in COLS]
        L.append(disp + " & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def single_label():
    SL_FULL = FULL
    ROWS = [("FedAvg", "FedAvg"), ("FedProx", "FedProx"), ("SCAFFOLD", "SCAFFOLD"),
            ("FedSAM", "FedSAM"), ("FedNTD (masking)", "FedNTD"),
            ("FedPrior (blend)", "FedPrior")]
    SL_COLS = [("C10-.1", "cifar_a0.1"), ("C10-.5", "cifar_a0.5"),
               ("C100-.1", "cifar100_a0.1"), ("C100-.5", "cifar100_a0.5")]

    def sms(tag, key):
        if key == "FedProx" and isinstance(RETUNE.get(tag), list):
            fs = RETUNE[tag]
            return statistics.mean(fs), statistics.pstdev(fs)
        m = _hist_ms(os.path.join(SL_FULL, f"results_{tag}.json"), key)
        return (m[0], m[1]) if m else None
    L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Single-label accuracy (\%)}, GroupNorm ResNet-18, "
         r"Dirichlet label skew, 3 seeds. The output-space mechanisms cluster with "
         r"FedAvg---\emph{except} that not-true masking (FedNTD) \emph{helps} under "
         r"extreme few-class skew (C10-.1), the sharp-not-true regime it was designed "
         r"for (Sec.~\ref{sec:why}); on multi-label it instead collapses. Balancing is "
         r"not applicable: softmax classes are mutually exclusive, so there is no "
         r"negative-weighting choice to make.}", r"\label{tab:sl}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         "Method & " + " & ".join(c for c, _ in SL_COLS) + r" \\", r"\midrule"]
    for disp, key in ROWS:
        cs = []
        for _, tag in SL_COLS:
            m = sms(tag, key)
            cs.append(f"{m[0]:.1f}\\tiny$\\pm${m[1]:.1f}" if m else "--")
        L.append(disp + " & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    pt = ml_table("pt_",
                  r"\textbf{Multi-label mAP (\%), ImageNet-pretrained backbone} "
                  r"(realistic regime), 3 seeds. Class-balancing "
                  r"the negatives (\textbf{BalBCE}) is the strongest lever; the dedicated "
                  r"methods FedMLP, FLAG, FedAwS omit it and none surpasses it; masking "
                  r"(FedNTD) collapses; blending (FedPrior) tracks FedAvg.", "tab:ml_pt", ML_ROWS)
    fs = ml_table("", r"Multi-label mAP (\%), from-scratch GroupNorm backbone, 3 seeds.",
                  "tab:ml_fs", ML_ROWS)
    block = "\n\n".join([pt, fs, twobytwo(), single_label(), ortho()]) + "\n"
    open(OUT, "w").write(block)
    print(block)
    print("AGG_C_DONE")


if __name__ == "__main__":
    main()
