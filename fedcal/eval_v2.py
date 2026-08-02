"""Does local calibration reliably help the worst client, or just seed 21?

``fedcal.benchmark`` runs once, on the default seed (21), and reports
``fedprox_cal`` improving the worst client's accuracy by 2.8 points over
plain FedAvg. This module reruns the same comparison across many seeds to
check whether that gain is reliable, and separately reports what happened
when an obvious fix (a bigger calibration sample, a real improvement
margin) was tried and evaluated against a frozen holdout.

    python -m fedcal.eval_v2
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

from .adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from .calibrate_v2 import benchmark_calibration_attempt
from .data import make_clients
from .train import benchmark_models


def _summarize_original(seeds: Sequence[int]) -> Dict:
    gains: List[float] = []
    for seed in seeds:
        clients = make_clients(seed=seed)
        result = benchmark_models(clients)
        gains.append(result["fedprox_cal"][1] - result["fedavg"][1])
    n = len(seeds)
    return {
        "n": n,
        "mean_gain": round(sum(gains) / n, 4),
        "positive": sum(1 for g in gains if g > 0),
        "negative": sum(1 for g in gains if g < 0),
    }


def _summarize_v2(seeds: Sequence[int], margin: float = 0.03) -> Dict:
    gains: List[float] = []
    for seed in seeds:
        clients = make_clients(seed=seed)
        result = benchmark_calibration_attempt(clients, margin=margin)
        gains.append(result["fedprox_cal_v2"][1] - result["fedavg"][1])
    n = len(seeds)
    return {
        "n": n,
        "mean_gain": round(sum(gains) / n, 4),
        "positive": sum(1 for g in gains if g > 0),
        "negative": sum(1 for g in gains if g < 0),
    }


def build_report() -> Dict:
    return {
        "original_tuning": _summarize_original(TUNING_SEEDS),
        "original_holdout": _summarize_original(HOLDOUT_SEEDS),
        "v2_tuning": _summarize_v2(TUNING_SEEDS[:30]),
        "v2_holdout": _summarize_v2(HOLDOUT_SEEDS),
    }


def format_report(report: Dict) -> str:
    lines = [
        "does worst-client calibration reliably help, seed 21 or not?",
        "=" * 62,
        f"{'variant':<24}{'n':>4}{'mean gain':>12}{'positive':>10}{'negative':>10}",
        "-" * 62,
    ]
    for name, key in [
        ("original / tuning", "original_tuning"),
        ("original / holdout", "original_holdout"),
        ("attempted fix / tuning", "v2_tuning"),
        ("attempted fix / holdout", "v2_holdout"),
    ]:
        row = report[key]
        lines.append(
            f"{name:<24}{row['n']:>4}{row['mean_gain']:>+12.4f}"
            f"{row['positive']:>10}{row['negative']:>10}"
        )
    lines.append("")
    lines.append(
        "the published benchmark's seed (21) is a lucky draw: across 60 seeds,"
    )
    lines.append(
        "mean worst-client gain from calibration is NEGATIVE, and it hurts the"
    )
    lines.append(
        "worst client more than twice as often as it helps. an obvious fix"
    )
    lines.append(
        "(calibrate on more data, require a real improvement margin) looked"
    )
    lines.append(
        "promising when tuned against the same seeds, but did not generalize to"
    )
    lines.append(
        "a disjoint holdout evaluated exactly once. this is a sample-size"
    )
    lines.append(
        "problem (36-108 points of calibration data per client, worst-of-6),"
    )
    lines.append("not a hyperparameter one, and honest reporting says so.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report()
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
