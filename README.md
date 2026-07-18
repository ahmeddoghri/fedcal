# fedcal

FedAvg's marketing photo has every client looking identical and happy. fedcal takes that photo, gives one client a bad day, and measures what happens.

![CI](https://github.com/ahmeddoghri/fedcal/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

FedAvg is easy to explain because every client politely disappears into one
neat average. Real clients do not disappear, they have different feature
shifts and different class skews, and somebody's device always ends up with
the worst slice of the data. fedcal makes that mess explicit: six clients
with different distributions, updates clipped so no single client can hijack
the global model, and a conservative local calibration step that only nudges
a client's bias when it actually improves validation accuracy, since letting
personalization run wild on a small local slice is how you overfit your way
into a worse model.

## Run it

```bash
git clone https://github.com/ahmeddoghri/fedcal
cd fedcal
pip install -e ".[dev]"
python -m fedcal.benchmark
```

## Verified benchmark

Generated locally with `python -m fedcal.benchmark`:

```text
model          macro_acc  worst_client_acc
fedavg            0.829             0.667
fedprox           0.833             0.667
fedprox_cal       0.815             0.694
worst_gain         0.028
```

Plain FedAvg and FedProx both strand the worst client at 66.7% accuracy while
looking fine on the macro average, which is exactly how a federated model
quietly fails one real user while the dashboard says everything is green.
Adding conservative local calibration trades 1.4 points of macro accuracy for
a 2.8 point gain on the worst client, on purpose, because the worst client is
the one who actually complains.

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
