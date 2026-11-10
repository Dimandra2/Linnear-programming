import numpy as np
import pandas as pd
import os


def generate_and_save_data(seed: int, add_trips=0):
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(seed)
    n_drivers = 6
    n_trips = 6 + add_trips

    drivers, trips = [], []
    for i in range(n_drivers):
        start = int(rng.integers(480, 1000))
        duration = int(rng.integers(120, 240))
        drivers.append({'id': f'D{i + 1}', 'start': start, 'end': start + duration})

    for j in range(n_trips):
        start = int(rng.integers(480, 1000))
        duration = int(rng.integers(120, 240))
        trips.append({'id': f'T{j + 1}', 'start': start, 'end': start + duration})

    for i in range(min(n_drivers, n_trips)):
        if drivers[i]['start'] > trips[i]['end'] or trips[i]['start'] > drivers[i]['end']:
            original_duration = trips[i]['end'] - trips[i]['start']
            new_start = drivers[i]['start'] + int(rng.integers(10, 30))
            trips[i]['start'] = new_start
            trips[i]['end'] = new_start + original_duration

    costs = []
    for i in range(n_drivers):
        for j in range(n_trips):
            if max(drivers[i]['start'], trips[j]['start']) < min(drivers[i]['end'], trips[j]['end']):
                costs.append({'driver': drivers[i]['id'], 'trip': trips[j]['id'], 'cost': int(rng.integers(10, 21))})

    pd.DataFrame(drivers).to_csv(f"data/drivers_{n_trips}t.csv", index=False)
    pd.DataFrame(trips).to_csv(f"data/trips_{n_trips}t.csv", index=False)
    pd.DataFrame(costs).to_csv(f"data/costs_{n_trips}t.csv", index=False)


def load_data(n_trips=6):
    drivers_df = pd.read_csv(f"data/drivers_{n_trips}t.csv")
    trips_df = pd.read_csv(f"data/trips_{n_trips}t.csv")
    costs_df = pd.read_csv(f"data/costs_{n_trips}t.csv")

    drivers = [(str(row['id']), int(row['start']), int(row['end'])) for _, row in drivers_df.iterrows()]
    trips = [(str(row['id']), int(row['start']), int(row['end'])) for _, row in trips_df.iterrows()]

    d_idx = {d[0]: i for i, d in enumerate(drivers)}
    t_idx = {t[0]: i for i, t in enumerate(trips)}

    costs = {}
    for _, row in costs_df.iterrows():
        costs[(int(d_idx[str(row['driver'])]), int(t_idx[str(row['trip'])]))] = int(row['cost'])
    return drivers, trips, costs