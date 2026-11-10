import argparse
import numpy as np
import time
from . import data_loader, model, solve, verify, heuristic, experiments


def main(seed: int):
    print(f"=== ЛР3. Планирование заказов (MILP). Seed: {seed} ===")

    # 1. Данные и Базовый MILP
    demand, unit_cost, fixed_cost, holding_cost = data_loader.generate_data(seed)
    c, constraints, integrality, bounds = model.build_model(demand, unit_cost, fixed_cost, holding_cost)

    res_milp = solve.solve_model(c, constraints, integrality, bounds)
    print(f"\n[1] MILP Статус: {res_milp.message}. Целевая функция: {res_milp.fun:.2f}")

    # Извлечение переменных
    n_items, n_periods = demand.shape
    x = np.zeros((n_items, n_periods))
    I = np.zeros((n_items, n_periods))
    y = np.zeros((n_items, n_periods))

    for i in range(n_items):
        for t in range(n_periods):
            idx = i * n_periods * 3 + t * 3
            x[i, t] = res_milp.x[idx]
            I[i, t] = res_milp.x[idx + 1]
            y[i, t] = round(res_milp.x[idx + 2])  # Округляем для проверки

    # 2. LP Релаксация
    res_lp = solve.solve_model(c, constraints, integrality, bounds, relax=True)
    gap = (res_milp.fun - res_lp.fun) / res_milp.fun * 100
    print(f"[2] LP-релаксация. Целевая функция: {res_lp.fun:.2f}. Интегральный зазор: {gap:.2f}%")

    # 3. Независимая проверка
    violations = verify.verify(demand, x, I, y, fixed_cost)
    if not violations:
        print("[3] Независимая проверка NumPy: ПРОЙДЕНА. Ограничения и Big-M соблюдены.")
    else:
        print(f"[3] Независимая проверка: ОШИБКИ! {violations}")

    # 4. Жадный алгоритм
    start_t = time.time()
    x_g, y_g, I_g, cost_g = heuristic.greedy_heuristic(demand, unit_cost, fixed_cost, holding_cost)
    greedy_time = time.time() - start_t
    gap_greedy = (cost_g - res_milp.fun) / res_milp.fun * 100
    print(f"\n[4] Жадный алгоритм: Стоимость {cost_g:.2f} у.е. (Зазор к MILP: +{gap_greedy:.2f}%)")
    print(f"    Время MILP: {res_milp.mip_node_count} итераций, Время Greedy: {greedy_time:.5f} сек.")

    # Запуск остальных обязательных экспериментов
    experiments.run_scenarios(seed)
    experiments.run_big_m_experiment(seed)
    experiments.run_scaling_experiment(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10109)
    args = parser.parse_args()
    main(args.seed)