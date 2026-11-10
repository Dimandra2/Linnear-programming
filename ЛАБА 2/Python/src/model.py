import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def build_model(drivers, trips, costs):
    # drivers: list of (id, start, end)
    # trips: list of (id, start, end)
    # costs: dict (i,j) -> cost, только для совместимых пар
    n_d = len(drivers)
    n_t = len(trips)
    # Индексация переменных: для каждой допустимой пары (i,j) — отдельная переменная
    var_index = {}
    idx = 0
    for i, (_, ds, de) in enumerate(drivers):
        for j, (_, ts, te) in enumerate(trips):
            if (i, j) in costs:  # дуга существует
                var_index[(i, j)] = idx
                idx += 1
    n_vars = idx
    c = np.zeros(n_vars)
    for (i, j), k in var_index.items():
        c[k] = costs[(i, j)]

    # Ограничения: sum_i x_ij = 1 для каждого рейса j
    # и sum_j x_ij <= 1 для каждого водителя i
    A_eq = lil_matrix((n_t, n_vars))
    b_eq = np.ones(n_t)
    for j in range(n_t):
        for i in range(n_d):
            if (i, j) in var_index:
                A_eq[j, var_index[(i, j)]] = 1

    A_ub = lil_matrix((n_d, n_vars))
    b_ub = np.ones(n_d)
    for i in range(n_d):
        for j in range(n_t):
            if (i, j) in var_index:
                A_ub[i, var_index[(i, j)]] = 1

    constraints = [
        LinearConstraint(A_eq.tocsr(), b_eq, b_eq),
        LinearConstraint(A_ub.tocsr(), -np.inf, b_ub)
    ]
    integrality = np.ones(n_vars)  # все переменные бинарные
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
    return c, constraints, integrality, bounds, var_index