import time
import matplotlib.pyplot as plt
import os
import numpy as np
from . import data_loader, model, solve, verify


def run_scaling_experiment(seed):
    """Эксперимент 4: Время решения / зазор при увеличении масштаба."""
    sizes = [5, 10, 15, 20]
    times = []
    gaps = []

    print("\n--- Эксперимент: Масштабирование (Кривая время/зазор) ---")
    for n in sizes:
        # Генерируем данные в памяти для конкретного размера
        demand, u_c, f_c, h_c = data_loader.generate_data(seed, n_items=n, n_periods=6)

        c, constraints, integrality, bounds = model.build_model(demand, u_c, f_c, h_c)

        start_time = time.time()
        res_milp = solve.solve_model(c, constraints, integrality, bounds)
        solve_time = time.time() - start_time

        res_lp = solve.solve_model(c, constraints, integrality, bounds, relax=True)

        gap = (res_milp.fun - res_lp.fun) / res_milp.fun * 100 if res_milp.success else 0
        times.append(solve_time)
        gaps.append(gap)
        print(f"Объектов: {n:2d} | Время MILP: {solve_time:.4f} с | Интегральный зазор: {gap:.2f}%")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o', color='b', label='Время решения MILP (сек)')
    plt.plot(sizes, gaps, marker='s', color='r', label='Интегральный зазор (%)')
    plt.xlabel("Количество товаров")
    plt.title("Кривая масштабирования MILP")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/scaling_curve.png", dpi=300)
    plt.close()
    print("[INFO] График масштабирования сохранен в results/scaling_curve.png")


def run_scenarios(seed):
    """Эксперимент 3: Сценарии (базовый, дефицитный, стрессовый)."""
    print("\n--- Эксперимент: Сценарии ---")
    scenarios = {}

    # Базовый (1.0)
    d_base, uc, fc, hc = data_loader.generate_data(seed, scale_factor=1.0)
    c, constr, integ, bnd = model.build_model(d_base, uc, fc, hc)
    res_base = solve.solve_model(c, constr, integ, bnd)
    scenarios['Базовый'] = res_base.fun

    # Дефицитный (рост спроса на 50%)
    d_def, _, _, _ = data_loader.generate_data(seed, scale_factor=1.5)
    c, constr, integ, bnd = model.build_model(d_def, uc, fc, hc)
    res_def = solve.solve_model(c, constr, integ, bnd)
    scenarios['Дефицитный (+50% спроса)'] = res_def.fun

    # Стрессовый (фикс. затраты x3)
    fc_stress = fc * 3
    c, constr, integ, bnd = model.build_model(d_base, uc, fc_stress, hc)
    res_stress = solve.solve_model(c, constr, integ, bnd)
    scenarios['Стрессовый (Фикс x3)'] = res_stress.fun

    for name, cost in scenarios.items():
        print(f"{name:30s} -> Целевая функция: {cost:.2f} у.е.")


def run_big_m_experiment(seed):
    """Эксперимент 6: Влияние размера Big-M на LP-релаксацию."""
    print("\n--- Эксперимент: Влияние Big-M ---")
    d_base, uc, fc, hc = data_loader.generate_data(seed)

    # Оптимальное (минимальное) M
    c, constr, integ, bnd = model.build_model(d_base, uc, fc, hc, use_big_m=True)
    res_lp_small = solve.solve_model(c, constr, integ, bnd, relax=True)
    res_milp = solve.solve_model(c, constr, integ, bnd)
    gap_small = (res_milp.fun - res_lp_small.fun) / res_milp.fun * 100

    # Неоправданно большое M
    c, constr, integ, bnd = model.build_model(d_base, uc, fc, hc, use_big_m=False)
    res_lp_large = solve.solve_model(c, constr, integ, bnd, relax=True)
    gap_large = (res_milp.fun - res_lp_large.fun) / res_milp.fun * 100

    print(
        f"С минимальным M (сумма спроса): LP = {res_lp_small.fun:.2f}, MILP = {res_milp.fun:.2f}, Gap = {gap_small:.2f}%")
    print(
        f"С огромным M (100000):          LP = {res_lp_large.fun:.2f}, MILP = {res_milp.fun:.2f}, Gap = {gap_large:.2f}%")