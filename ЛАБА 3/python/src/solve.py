from scipy.optimize import milp


def solve_model(c, constraints, integrality, bounds, relax=False):
    # Отключаем целочисленность, если нужна LP-релаксация
    int_cond = None if relax else integrality

    res = milp(c=c, constraints=constraints, integrality=int_cond, bounds=bounds,
               options={'disp': False, 'time_limit': 30})
    return res