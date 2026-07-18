from __future__ import annotations

import math

from .data import ClientData, Point
from .model import accuracy, macro_accuracy, prob


def _clip_update(delta: list[float], clip: float) -> list[float]:
    norm = math.sqrt(sum(value * value for value in delta))
    if norm <= clip:
        return delta
    scale = clip / max(norm, 1e-12)
    return [value * scale for value in delta]


def _local_train(
    weights: list[float],
    rows: list[Point],
    global_weights: list[float],
    epochs: int,
    lr: float,
    prox_mu: float,
) -> list[float]:
    w = weights[:]
    for _ in range(epochs):
        for point in rows:
            p = prob(w, point)
            err = p - point.y
            grad = [err, err * point.x1, err * point.x2]
            for idx in range(3):
                grad[idx] += prox_mu * (w[idx] - global_weights[idx])
                w[idx] -= lr * grad[idx]
    return w


def fedavg(
    clients: list[ClientData],
    rounds: int = 28,
    epochs: int = 2,
    lr: float = 0.05,
    prox_mu: float = 0.0,
    clip: float = 1.8,
) -> list[float]:
    global_w = [0.0, 0.0, 0.0]
    for _ in range(rounds):
        updates = []
        total = 0
        for client in clients:
            local = _local_train(global_w, client.train, global_w, epochs, lr, prox_mu)
            delta = _clip_update([local[i] - global_w[i] for i in range(3)], clip)
            updates.append((len(client.train), delta))
            total += len(client.train)
        for idx in range(3):
            global_w[idx] += sum(weight * delta[idx] for weight, delta in updates) / total
    return global_w


def personalize_bias(weights: list[float], client: ClientData) -> float:
    best_offset = 0.0
    best_acc = accuracy(weights, client.val, 0.0)
    for step in range(-20, 21):
        offset = step * 0.08
        acc = accuracy(weights, client.val, offset)
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_offset = offset
    return best_offset


def benchmark_models(clients: list[ClientData]) -> dict[str, tuple[float, float]]:
    avg = fedavg(clients, prox_mu=0.0)
    prox = fedavg(clients, prox_mu=0.10)
    offsets = [personalize_bias(prox, client) for client in clients]
    return {
        "fedavg": macro_accuracy(avg, [(client.test, 0.0) for client in clients]),
        "fedprox": macro_accuracy(prox, [(client.test, 0.0) for client in clients]),
        "fedprox_cal": macro_accuracy(prox, [(client.test, offset) for client, offset in zip(clients, offsets)]),
    }
