from __future__ import annotations

import math

from .data import Point


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score(weights: list[float], point: Point, bias_offset: float = 0.0) -> float:
    return weights[0] + bias_offset + weights[1] * point.x1 + weights[2] * point.x2


def prob(weights: list[float], point: Point, bias_offset: float = 0.0) -> float:
    return sigmoid(score(weights, point, bias_offset))


def accuracy(weights: list[float], rows: list[Point], bias_offset: float = 0.0) -> float:
    correct = 0
    for point in rows:
        pred = 1 if prob(weights, point, bias_offset) >= 0.5 else 0
        correct += int(pred == point.y)
    return correct / len(rows)


def macro_accuracy(weights: list[float], client_rows: list[tuple[list[Point], float]]) -> tuple[float, float]:
    scores = [accuracy(weights, rows, offset) for rows, offset in client_rows]
    return sum(scores) / len(scores), min(scores)
