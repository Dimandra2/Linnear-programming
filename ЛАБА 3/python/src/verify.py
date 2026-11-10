import numpy as np


def verify(demand, x, I, y, fixed_cost, tolerance=1e-5):
    n_items, n_periods = demand.shape
    violations = []

    # 1. Проверка баланса
    for i in range(n_items):
        prev_I = 0
        for t in range(n_periods):
            current_balance = prev_I + x[i, t] - I[i, t]
            if abs(current_balance - demand[i, t]) > tolerance:
                violations.append(f"Нарушение баланса: Товар {i}, Период {t}")
            prev_I = I[i, t]

    # 2. Проверка Big-M (связка заказа и бинарной переменной)
    for i in range(n_items):
        for t in range(n_periods):
            if x[i, t] > tolerance and y[i, t] < 0.99:
                violations.append(f"Нарушение Big-M: Заказ есть (x={x[i, t]}), но y={y[i, t]}")

    # 3. Проверка бинарности
    if np.any((y < -tolerance) | (y > 1 + tolerance)):
        violations.append("Нарушена бинарность переменных y")

    return violations