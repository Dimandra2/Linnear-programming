import numpy as np
import pandas as pd
import os


def generate_data(seed: int, n_items=6, n_periods=6, scale_factor=1.0):
    """Генерирует данные для задачи планирования заказов."""
    rng = np.random.default_rng(seed)

    items = [f"Item_{i + 1}" for i in range(n_items)]
    periods = [f"T_{t + 1}" for t in range(n_periods)]

    # Случайные спросы (увеличиваем на scale_factor для сценариев)
    demand = rng.integers(10, 50, size=(n_items, n_periods)) * scale_factor
    demand = np.ceil(demand).astype(int)

    # Затраты на закупку (за единицу)
    unit_cost = rng.integers(5, 15, size=(n_items, n_periods))

    # Затраты на хранение (за единицу за период)
    holding_cost = rng.integers(1, 4, size=n_items)

    # Фиксированные затраты на заказ
    fixed_cost = rng.integers(50, 100, size=(n_items, n_periods))

    # Сохранение в CSV для требования "данные отдельно"
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(demand, index=items, columns=periods).to_csv("data/demand.csv")
    pd.DataFrame(unit_cost, index=items, columns=periods).to_csv("data/unit_cost.csv")
    pd.DataFrame(fixed_cost, index=items, columns=periods).to_csv("data/fixed_cost.csv")
    pd.DataFrame(holding_cost, index=items).to_csv("data/holding_cost.csv")

    return demand, unit_cost, fixed_cost, holding_cost


def load_data():
    demand = pd.read_csv("data/demand.csv", index_col=0).values
    unit_cost = pd.read_csv("data/unit_cost.csv", index_col=0).values
    fixed_cost = pd.read_csv("data/fixed_cost.csv", index_col=0).values
    holding_cost = pd.read_csv("data/holding_cost.csv", index_col=0).values.flatten()
    return demand, unit_cost, fixed_cost, holding_cost