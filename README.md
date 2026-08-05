# fedcal

FedAvg's marketing photo has every client looking identical and happy. fedcal takes that photo, gives one client a bad day, and measures what happens.

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

**Update:** that 2.8 point gain is one lucky seed. Across 60 seeds, mean
worst-client gain from calibration is *negative* (-2.1 points), and it
hurts the worst client more than twice as often as it helps (28 seeds
worse, 13 better). An obvious fix looked promising when tuned, then
failed on a frozen holdout evaluated exactly once. `python -m
fedcal.eval_v2` runs the honest multi-seed comparison. Details below.

## Local calibration usually hurts the worst client, not helps it

`personalize_bias` picks the bias offset that maximizes accuracy on a
client's validation split, 36 points at the default client size, and
applies it to that client's test predictions. The README calls this
conservative: it only applies an offset "when it actually improves
validation accuracy." Measured across 60 seeds, that framing is
backwards for the metric this whole benchmark is built around:

```bash
python -m fedcal.eval_v2
```
```
variant                    n   mean gain  positive  negative
original / tuning         60     -0.0213        13        28
original / holdout        20     -0.0222         5        12
attempted fix / tuning    30     +0.0046        10         3
attempted fix / holdout   20     -0.0111         3         8
```

Mean worst-client gain from calibration is negative, and it hurts the
worst client more than twice as often as it helps. Seed 21, the one in
the published table, is one of the 13 lucky draws out of 60.

I tried the obvious fix: calibrate on train+val combined (108 points
instead of 36) and only apply an offset if it clears a real improvement
margin instead of any positive epsilon, so a single noisy validation
point can't trigger a change. Tuned against 30 seeds, it looked like it
worked (mean gain +0.46 points, 10 positive vs 3 negative). Evaluated
exactly once against a disjoint 20-seed holdout, it didn't hold up (mean
gain back to -1.11 points, 3 positive vs 8 negative).

That failure is reported here instead of hidden, because it's the more
useful result. It means the instability isn't a hyperparameter you can
tune away; it's a sample-size problem. With only 6 clients and 36-108
points of calibration data each, worst-client accuracy moves by roughly
2.8 percentage points per flipped test example, and no offset-selection
rule built on that little data reliably beats noise. `train.py` and
`benchmark.py` are untouched, and the published table above still
reproduces exactly; `calibrate_v2.py` and `eval_v2.py` document the
attempt and its honest result rather than claiming a fix that doesn't
generalize.

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
