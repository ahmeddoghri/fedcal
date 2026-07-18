"""Federated learning with non-IID clients and local calibration."""

from .data import ClientData, Point, make_clients
from .train import benchmark_models, fedavg, personalize_bias

__all__ = ["ClientData", "Point", "benchmark_models", "fedavg", "make_clients", "personalize_bias"]
