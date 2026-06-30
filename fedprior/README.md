# FedPrior

A clean, modular re-implementation of the **FedPrior** algorithm — a federated
learning method that treats the global model as **prior information** in the
sense of the *prior LASSO* (pLasso) method, and the four baselines it is
compared against.

## The idea in one line

> **FedProx** regularises in *parameter* space (`½·μ·‖w − w₀‖²`).
> **FedPrior** regularises in *output / response* space: each client is pulled
> toward the **global model's predictions** on its own data — exactly the
> prior-LASSO philosophy of injecting prior information through the response,
> not the coefficients.

Local objective for a client with global (prior) model `w₀`:

```
L_FedPrior(w) = CE(f_w(x), y)  +  η · KD( f_w(x) ‖ f_{w₀}(x) )
```

where `KD` is a soft-distillation term (temperature `T`) toward the frozen
global model's softmax distribution. With `not_true=True` the distillation runs
only over the non-ground-truth classes (FedNTD-style), preserving global
knowledge without re-teaching labels the client already owns.

`PriorProx` combines both terms: `CE + ½·μ·‖w−w₀‖² + η·KD`, applied to every
participating client.

## Layout

```
fedprior/
  models.py       LinearNet (softmax, bias absorbed by a constant feature) + Client
  data.py         FedProx synthetic(alpha,beta) generator + client/test split
  algorithms.py   LocalTrainer + Global/FedAvg/FedProx/FedPrior/PriorProx
  run.py          ExperimentConfig, run_experiment(), plot_history(), CLI
```

## Quick start

```bash
# fast sanity run
python -m fedprior.run --alpha 1 --beta 1 --rounds 20 --straggler 0.5

# FedPrior with not-true distillation + adaptive eta
python -m fedprior.run --rounds 200 --not-true --adaptive-eta
```

```python
from fedprior import ExperimentConfig, TrainConfig, run_experiment, plot_history

cfg = ExperimentConfig(
    alpha=1, beta=1, rounds=200,
    train=TrainConfig(straggler_rate=0.5, eta=0.1, not_true=True),
)
results = run_experiment(cfg)          # {method: (weight, history)}
plot_history(results, metric="test_acc")
```

Each `history` is a dict with `loss`, `train_acc`, `test_acc`, `coef`
(the live μ or η per round).

## What changed vs. the original notebooks

**Faithful-pLasso upgrades**
- Soft prior distribution instead of hard `argmax` pseudo-labels.
- Optional not-true (FedNTD-style) masking.
- Adaptive η (reuses the FedProx dynamic-coefficient heuristic).

**Bug fixes**
- FedProx proximal term used `weight.detach()` inside the norm → **zero
  gradient** (the term was inert). Now regularises the live weight.
- `PriorProx` applied the prior term to stragglers only; active clients silently
  fell back to plain training. Now consistent across all participants.
- `evaluate` no longer mutates client tensors in place (GPU side effect).
- Synthetic generator now labels **all** sampled points (the original left half
  of each user's points labelled `0`) and splits train/test 50/50.
- Removed the stray hard-coded `FedAvg(25, …)` round count.
- All randomness is seeded for reproducibility.

## Note on scope

This is a **convex linear softmax** benchmark. It is faithful to the original
work and good for sanity/ablation, but client-drift effects (where FedPrior /
FedNTD / SCAFFOLD actually separate from FedAvg) only emerge on **non-convex**
models. To make a competitive claim, add a small CNN on MNIST/CIFAR with a
Dirichlet non-IID split and include FedNTD as a baseline — see the discussion in
the chat handoff.
