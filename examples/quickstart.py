from fedcal import benchmark_models, make_clients

metrics = benchmark_models(make_clients(clients=4, n=120))
print(f"fedprox_cal_macro={metrics['fedprox_cal'][0]:.3f}")
