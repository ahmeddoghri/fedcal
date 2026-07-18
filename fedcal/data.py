from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x1: float
    x2: float
    y: int


@dataclass(frozen=True)
class ClientData:
    name: str
    train: list[Point]
    val: list[Point]
    test: list[Point]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_points(rng: random.Random, n: int, shift: float, bias: float) -> list[Point]:
    rows = []
    for _ in range(n):
        x1 = rng.gauss(shift, 1.0)
        x2 = rng.gauss(-0.5 * shift, 1.0)
        logit = 1.3 * x1 - 1.0 * x2 + bias
        y = 1 if rng.random() < _sigmoid(logit) else 0
        rows.append(Point(x1, x2, y))
    return rows


def make_clients(seed: int = 21, clients: int = 6, n: int = 180) -> list[ClientData]:
    rng = random.Random(seed)
    out = []
    for idx in range(clients):
        shift = -1.2 + idx * (2.4 / max(1, clients - 1))
        bias = -0.9 + idx * (1.8 / max(1, clients - 1))
        rows = _make_points(rng, n, shift, bias)
        out.append(
            ClientData(
                name=f"client_{idx}",
                train=rows[: int(0.6 * n)],
                val=rows[int(0.6 * n) : int(0.8 * n)],
                test=rows[int(0.8 * n) :],
            )
        )
    return out
