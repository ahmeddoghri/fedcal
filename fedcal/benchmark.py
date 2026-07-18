from __future__ import annotations

from .data import make_clients
from .train import benchmark_models


def main() -> None:
    rows = benchmark_models(make_clients())
    print("fedcal benchmark: non-IID federated learning")
    print("model          macro_acc  worst_client_acc")
    for label, (macro, worst) in rows.items():
        print(f"{label:13s} {macro:9.3f}  {worst:16.3f}")
    print(f"worst_gain     {rows['fedprox_cal'][1] - rows['fedavg'][1]:9.3f}")


if __name__ == "__main__":
    main()
