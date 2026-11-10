import numpy as np

def verify_solution(x, drivers, trips, costs, var_index):
    n_d = len(drivers)
    n_t = len(trips)
    # Восстанавливаем матрицу назначений
    assign = np.zeros((n_d, n_t))
    for (i, j), k in var_index.items():
        assign[i, j] = x[k]
    # 1. Каждый рейс покрыт ровно одним водителем
    trip_cover = assign.sum(axis=0)
    assert np.allclose(trip_cover, 1.0), "Нарушено покрытие рейсов"
    # 2. Каждый водитель назначен не более чем на один рейс
    driver_load = assign.sum(axis=1)
    assert np.all(driver_load <= 1.0 + 1e-9), "Водитель перегружен"
    # 3. Проверка границ (0/1)
    assert np.all((assign >= -1e-9) & (assign <= 1 + 1e-9)), "Нарушены границы"
    # 4. Пересчёт целевой функции
    obj = sum(costs[(i, j)] * assign[i, j] for (i, j) in var_index)
    return {
        'feasible': True,
        'objective': obj,
        'trip_cover': trip_cover.tolist(),
        'driver_load': driver_load.tolist()
    }