from scipy.optimize import milp
import numpy as np

def solve_model(c, constraints, integrality, bounds):
    options = {
        'disp': True,          # вывод лога солвера
        'time_limit': 60,      # секунд
        'mip_rel_gap': 0.0     # требовать оптимальность
    }
    res = milp(c=c, constraints=constraints, integrality=integrality,
               bounds=bounds, options=options)
    return res