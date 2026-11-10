import numpy as np


def greedy_heuristic(demand, unit_cost, fixed_cost, holding_cost):
    """Жадный алгоритм: заказываем ровно столько, сколько нужно."""
    n_items, n_periods = demand.shape
    x = np.zeros_like(demand)
    y = np.zeros_like(demand)
    I = np.zeros_like(demand)
    total_cost = 0.0

    for i in range(n_items):
        prev_I = 0
        for t in range(n_periods):
            need = demand[i, t] - prev_I
            if need > 0:
                x[i, t] = need
                y[i, t] = 1
                total_cost += fixed_cost[i, t] + need * unit_cost[i, t]
                I[i, t] = 0
                prev_I = 0
            else:
                x[i, t] = 0
                y[i, t] = 0
                I[i, t] = -need
                total_cost += -need * holding_cost[i]
                prev_I = I[i, t]

    return x, y, I, total_cost