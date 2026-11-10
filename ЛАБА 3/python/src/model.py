import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import LinearConstraint, Bounds


def build_model(demand, unit_cost, fixed_cost, holding_cost, use_big_m=True):
    n_items, n_periods = demand.shape
    # Переменные: x (заказ), I (запас), y (бинарная на заказ)
    # Индексы: x[i,t], I[i,t], y[i,t]

    n_vars = n_items * n_periods * 3
    c = np.zeros(n_vars)

    # Заполнение целевой функции
    for i in range(n_items):
        for t in range(n_periods):
            idx_x = i * n_periods * 3 + t * 3
            idx_I = idx_x + 1
            idx_y = idx_x + 2
            c[idx_x] = unit_cost[i, t]
            c[idx_I] = holding_cost[i]
            c[idx_y] = fixed_cost[i, t]

    # Ограничения баланса: I[i,t] - I[i,t-1] - x[i,t] = -demand[i,t]
    # Для t=0: I[i,0] - x[i,0] = -demand[i,0]
    A_eq = lil_matrix((n_items * n_periods, n_vars))
    b_eq = np.zeros(n_items * n_periods)

    for i in range(n_items):
        for t in range(n_periods):
            row = i * n_periods + t
            idx_x = i * n_periods * 3 + t * 3
            idx_I = idx_x + 1
            idx_y = idx_x + 2

            A_eq[row, idx_I] += 1
            A_eq[row, idx_x] -= 1
            b_eq[row] = -demand[i, t]

            if t > 0:
                idx_I_prev = idx_I - 3
                A_eq[row, idx_I_prev] -= 1

    # Ограничения Big-M: x[i,t] - M[i,t] * y[i,t] <= 0
    A_ub = lil_matrix((n_items * n_periods, n_vars))
    b_ub = np.zeros(n_items * n_periods)

    for i in range(n_items):
        for t in range(n_periods):
            row = i * n_periods + t
            idx_x = i * n_periods * 3 + t * 3
            idx_y = idx_x + 2

            # Выбор Big-M (если выключен, берем очень большое)
            if use_big_m:
                M = sum(demand[i, t:])  # Минимально необходимое M
            else:
                M = 100000

            A_ub[row, idx_x] += 1
            A_ub[row, idx_y] -= M

    constraints = [
        LinearConstraint(A_eq.tocsr(), b_eq, b_eq),
        LinearConstraint(A_ub.tocsr(), -np.inf, b_ub)
    ]

    # Бинарность только для y (индекс +2)
    integrality = np.zeros(n_vars)
    for i in range(n_items * n_periods):
        integrality[i * 3 + 2] = 1

    bounds = Bounds(np.zeros(n_vars), np.full(n_vars, np.inf))
    for i in range(n_items * n_periods):
        bounds.ub[i * 3 + 2] = 1.0  # y <= 1

    return c, constraints, integrality, bounds