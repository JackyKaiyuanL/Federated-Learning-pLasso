# Reference implementation (anonymous)

Code for the FLTA 2026 submission *"Dissecting Output-Space Methods for Multi-Label
Federated Learning: A Controlled Study."* Anonymous release for double-blind review.

## What the paper claims (read this first)
A controlled study, not a new method. In multi-label federated learning the lever is
**class-balanced BCE** (BalBCE: positive and negative BCE each averaged over their own
entries, then summed), not the federated output-space machinery layered on top. The
finding is scoped: balancing dominates in the pretrained-backbone regime and on COCO,
while on from-scratch VOC the lever is negative distillation instead. Balancing is shown
to be **orthogonal and compatible** with the dedicated methods rather than a competitor.

## Method registry (`fedprior/algorithms.py`)
Parameter-space baselines: `FedAvg`, `FedProx`, `SCAFFOLD`, `FedSAM`.
Output-space / multi-label: `FedNTD` (not-true masking), `FedAwS`, `FedMLP`, `FLAG`.
The lever and controls: `BalBCE` (class-balanced BCE); `FedNTD-BCE` (balanced BCE +
negative distillation, shown as **FedND** in the paper); `FedND-std` (negatives restored
in standard/unbalanced form, the clean causal control); `FedPrior` (the response blend,
eta=0.1). `FedMLP-bal` is FedMLP with class-balanced BCE (the orthogonality check).

## Which script produces which result
| paper item | script |
|---|---|
| Multi-label tables (pretrained + from-scratch, incl. FLAG) and the 2x2 dissection | `experiments/full_benchmark.py` -> `experiments/aggregate_c.py` |
| Single-label CIFAR-10/100 table | `experiments/full_benchmark.py` -> `experiments/aggregate_sl.py` |
| Orthogonality table (+balanced BCE into FedMLP/FedNTD) | `experiments/b2_orthogonality.py` (results in `results_b2/`) |
| Score-margin figure (5 curves incl. the FedND-std causal control) | `experiments/diag_m4.py` (uses `results_diag_m4.json`) |

## Reproduce
```bash
conda create -n fedprior python=3.12 -y && conda activate fedprior
pip install -r requirements.txt          # torch (CUDA build), torchvision, numpy, scikit-learn, matplotlib

# Full bake-off (3 seeds) -> writes experiments/full/*.json:
PYTHONPATH=. python -m experiments.full_benchmark --tasks synthetic cifar cifar100 voc coco --alphas 0.1 0.5
# Tables (reads experiments/full/*.json and the shipped results_b2/*.json):
PYTHONPATH=. python -m experiments.aggregate_c
PYTHONPATH=. python -m experiments.aggregate_sl
# Orthogonality runs (pretrained backbone, VOC+COCO):
PYTHONPATH=. python -m experiments.b2_orthogonality
# Score-margin figure (adds the FedND-std curve):
PYTHONPATH=. python -m experiments.diag_m4
```
The aggregation scripts read the bake-off results from `experiments/full/`; set the
environment variable `FEDPRIOR_FULL` to point elsewhere if you keep them in another
location. `results_b2/` and `results_diag_m4.json` are shipped so the orthogonality and
margin-control results can be inspected without a full re-run.

## Data
`./data` is auto-created. CIFAR-10/100 and VOC2007 download on first use. MS-COCO must be
placed at `./data/coco/annotations/instances_val2017.json` + `./data/coco/val2017/`. COCO
here is the val2017 subset: train and test are an 80/20 split of the annotated val2017
images (fixed seed), ~4k train / ~1k test.

## Notes
- GroupNorm (not BatchNorm) in the from-scratch ResNet: BN running stats do not average
  meaningfully across clients.
- Multi-label metric is **macro mAP** (`average_precision_score` per class, averaged over
  classes with at least one positive in the test set).
- Deep tasks: cosine LR (floored at 1% of base), 3 seeds, last-10-round mean reported.
- Pretrained backbone: ImageNet ResNet-18 with frozen BatchNorm.
