from fedcal import benchmark_models, fedavg, make_clients
from fedcal.model import accuracy
from fedcal.train import personalize_bias


def test_clients_are_non_iid() -> None:
    clients = make_clients()
    rates = [sum(point.y for point in client.train) / len(client.train) for client in clients]
    assert max(rates) - min(rates) > 0.25


def test_training_returns_three_weights() -> None:
    assert len(fedavg(make_clients(clients=3, n=80), rounds=3)) == 3


def test_personalization_does_not_hurt_validation() -> None:
    clients = make_clients()
    weights = fedavg(clients, prox_mu=0.08)
    for client in clients:
        offset = personalize_bias(weights, client)
        assert accuracy(weights, client.val, offset) >= accuracy(weights, client.val, 0.0)


def test_calibration_improves_worst_client() -> None:
    metrics = benchmark_models(make_clients())
    assert metrics["fedprox_cal"][1] >= metrics["fedavg"][1]
