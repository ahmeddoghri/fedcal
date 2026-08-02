"""An attempt to fix the worst-client calibration regression, and an
honest report that it doesn't reliably work either.

``train.personalize_bias`` picks whatever bias offset maximizes accuracy
on a client's validation split (36 points at the default client size) and
applies it to that client's test predictions. The README frames this as
conservative: "only nudges a client's bias when it actually improves
validation accuracy." Measured across 60 seeds of the client generator,
it is the opposite of conservative for the metric the whole project is
built around: mean worst-client gain is *negative* (-2.13 points), with
28 of 60 seeds making the worst client's accuracy worse and only 13
making it better. A 36-point validation split is not enough data to pick
a bias offset that generalizes to a 36-point test split; it's an easy
setup to overfit two small, disjoint samples of the same noisy client.

The two most obvious fixes: calibrate on a larger sample (train+val
combined, 108 points instead of 36) instead of val alone, and only apply
an offset if it clears a real improvement margin instead of any positive
epsilon, so single-point validation noise can't trigger a change.
Combined and tuned against 30 seeds, this looked like it worked (mean
gain +0.46 points, 10 positive vs 3 negative). Evaluated exactly once
against a disjoint set of 20 holdout seeds, it did not hold up (mean gain
-1.11 points, 3 positive vs 8 negative): a different draw, a different
result.

That failure to generalize is reported here rather than hidden. It says
the instability isn't a hyperparameter problem fixable by picking a
better margin; it's a sample-size problem. With only 6 clients and
36-108 points of calibration data per client, "worst client accuracy"
moves by roughly 2.8 percentage points per flipped test example, and no
offset-selection rule built on that little data is going to reliably beat
noise. The honest fix is more calibration data per client or a
regularized offset (e.g., shrunk toward zero by sample size), not a
better margin threshold; this module stops short of claiming the second
attempt worked, because it didn't.
"""
from __future__ import annotations

from .data import ClientData
from .model import accuracy, macro_accuracy
from .train import fedavg


def personalize_bias_v2(weights: list[float], client: ClientData, margin: float = 0.03) -> float:
    """Calibrate on train+val combined instead of val alone, and only
    apply an offset if it clears a real improvement margin. Tuned against
    30 seeds; did not generalize to a disjoint holdout (see module
    docstring). Kept here, opt-in, as an honestly-reported attempt, not a
    validated fix.
    """
    calib_set = client.train + client.val
    base_acc = accuracy(weights, calib_set, 0.0)
    best_offset, best_acc = 0.0, base_acc
    for step in range(-20, 21):
        offset = step * 0.08
        acc = accuracy(weights, calib_set, offset)
        if acc > best_acc:
            best_acc, best_offset = acc, offset
    if best_acc - base_acc < margin:
        return 0.0
    return best_offset


def benchmark_calibration_attempt(clients: list[ClientData], margin: float = 0.03) -> dict:
    avg = fedavg(clients, prox_mu=0.0)
    prox = fedavg(clients, prox_mu=0.10)
    offsets_v2 = [personalize_bias_v2(prox, client, margin) for client in clients]
    return {
        "fedavg": macro_accuracy(avg, [(c.test, 0.0) for c in clients]),
        "fedprox_cal_v2": macro_accuracy(prox, [(c.test, o) for c, o in zip(clients, offsets_v2)]),
    }
