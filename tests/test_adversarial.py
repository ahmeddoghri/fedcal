"""Tests for the worst-client calibration regression finding."""

from __future__ import annotations

from fedcal.adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from fedcal.data import make_clients
from fedcal.eval_v2 import _summarize_original, _summarize_v2, build_report
from fedcal.train import benchmark_models

# --- the finding: the published seed is a lucky draw -----------------------

def test_published_seed_shows_a_positive_gain():
    """Confirm the exact published-benchmark result: at the default seed
    (21), calibration does improve the worst client."""
    metrics = benchmark_models(make_clients())
    assert metrics["fedprox_cal"][1] > metrics["fedavg"][1]


def test_calibration_mean_gain_is_negative_across_many_seeds():
    """The claim "conservative calibration only nudges when it actually
    improves validation accuracy" implies it shouldn't hurt the worst
    client on average. It does."""
    result = _summarize_original(TUNING_SEEDS)
    assert result["mean_gain"] < 0
    assert result["negative"] > result["positive"]


def test_calibration_hurts_the_worst_client_more_often_than_it_helps():
    result = _summarize_original(TUNING_SEEDS)
    assert result["negative"] >= 2 * result["positive"]


# --- an attempted fix, honestly reported as not generalizing ---------------

def test_attempted_fix_looks_better_on_the_tuning_seeds():
    """The margin + larger-calibration-set fix does look like an
    improvement on the seeds it was tuned against."""
    result = _summarize_v2(TUNING_SEEDS[:30])
    assert result["mean_gain"] > 0
    assert result["positive"] > result["negative"]


def test_attempted_fix_does_not_generalize_to_the_frozen_holdout():
    """The same fix, evaluated exactly once on a disjoint holdout, goes
    back to a negative mean gain -- the improvement did not generalize,
    and that failure is the honest result to report."""
    result = _summarize_v2(HOLDOUT_SEEDS)
    assert result["mean_gain"] < 0


def test_holdout_seeds_are_disjoint_from_tuning_seeds():
    assert not (set(TUNING_SEEDS) & set(HOLDOUT_SEEDS))


def test_original_calibration_also_underperforms_on_the_same_holdout():
    """The holdout isn't just harder for the attempted fix; the original
    calibration approach is negative there too, consistent with the
    tuning-set finding rather than a fluke of the fix attempt."""
    result = _summarize_original(HOLDOUT_SEEDS)
    assert result["mean_gain"] < 0


# --- the original module is untouched ---------------------------------------

def test_original_train_module_untouched():
    import fedcal.train as train_module

    assert not hasattr(train_module, "personalize_bias_v2")


def test_original_benchmark_still_reproduces():
    clients = make_clients()
    metrics = benchmark_models(clients)
    fedavg_macro, fedavg_worst = metrics["fedavg"]
    fedprox_macro, fedprox_worst = metrics["fedprox"]
    cal_macro, cal_worst = metrics["fedprox_cal"]
    assert round(fedavg_macro, 3) == 0.829
    assert round(fedavg_worst, 3) == 0.667
    assert round(fedprox_macro, 3) == 0.833
    assert round(cal_macro, 3) == 0.815
    assert round(cal_worst, 3) == 0.694


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report() == build_report()
