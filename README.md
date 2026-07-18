# fedcal

A small federated learning benchmark for the thing that breaks pretty demos:
non-IID clients. It trains logistic regression with FedAvg, adds a FedProx
penalty, clips client updates, and calibrates a local client bias on held-out
validation data.

![CI](https://github.com/ahmeddoghri/fedcal/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

## Run it

```bash
git clone https://github.com/ahmeddoghri/fedcal
cd fedcal
pip install -e ".[dev]"
python -m fedcal.benchmark
```

## Verified benchmark

These numbers were generated locally with `python -m fedcal.benchmark`:

```text
model          macro_acc  worst_client_acc
fedavg            0.829             0.667
fedprox           0.833             0.667
fedprox_cal       0.815             0.694
worst_gain         0.028
```

## Research trail

- Federated learning on non-IID data survey, 2024: https://arxiv.org/abs/2411.12377
- Personalized federated learning via feature distribution adaptation, 2024: https://arxiv.org/abs/2411.00329
- Personalized federated learning on flowing data heterogeneity, 2024: https://arxiv.org/html/2410.01502v1
- Differentially private federated learning systematic review, 2024: https://arxiv.org/abs/2405.08299

## Tests

```bash
pytest -q
ruff check .
```

MIT © Ahmed Doghri
