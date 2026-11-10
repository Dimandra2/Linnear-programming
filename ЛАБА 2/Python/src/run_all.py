import argparse
import numpy as np
import pandas as pd
import json
import traceback

from . import data_loader, model, solve, verify, greedy, visualizer


def solve_and_verify(drivers, trips, costs):
    c, constraints, integrality, bounds, var_index = model.build_model(drivers, trips, costs)
    res = solve.solve_model(c, constraints, integrality, bounds)
    if res.success:
        check = verify.verify_solution(np.round(res.x), drivers, trips, costs, var_index)
        return res, var_index, check
    return res, var_index, None


def main(seed: int):
    print(f"=== Запуск пайплайна с seed: {seed} ===")
    try:
        # 1. Данные
        data_loader.generate_and_save_data(seed, add_trips=0)
        drivers, trips, costs = data_loader.load_data(n_trips=6)
        print("[INFO] Данные загружены из data/*.csv")

        # 2. Решение
        res, var_index, check = solve_and_verify(drivers, trips, costs)
        if check is None:
            print(f"ОШИБКА: Решение не найдено. Статус: {res.message}")
            return
        print(f"Базовое решение MILP: {check['objective']} у.е. (Проверка пройдена)")

        # Сохранение
        with open("results/verification.json", "w") as f:
            json.dump(check, f, indent=4)
        sol_rows = [{'Driver': drivers[i][0], 'Trip': trips[j][0], 'Cost': costs[(i, j)]}
                    for (i, j), k in var_index.items() if round(res.x[k]) == 1]
        pd.DataFrame(sol_rows).to_csv("results/solution.csv", index=False)

        # 3. Визуализация
        visualizer.draw_network(drivers, trips, costs, {k: res.x[idx] for k, idx in var_index.items()})
        visualizer.draw_heatmap(drivers, trips, costs)

        # 4. Жадный алгоритм
        greedy_assign, greedy_total = greedy.smart_greedy(drivers, trips, costs)
        if greedy_assign:
            print(
                f"Жадный алгоритм: {greedy_total} у.е. (Зазор: {((greedy_total - check['objective']) / check['objective']) * 100:.2f}%)")
        else:
            print("Жадный алгоритм не нашел решения (Infeasible).")

        # 5. Сценарии
        scenarios = [{'Scenario': 'Base', 'Objective': check['objective'], 'Status': 'Optimal'}]

        # Отказ дуги
        worst_arc = max(sol_rows, key=lambda x: x['Cost'])
        d_i = [d[0] for d in drivers].index(worst_arc['Driver'])
        t_j = [t[0] for t in trips].index(worst_arc['Trip'])
        costs_broken = costs.copy()
        del costs_broken[(d_i, t_j)]

        res_b, _, check_b = solve_and_verify(drivers, trips, costs_broken)
        if res_b.success:
            print(f"[Сценарий Отказ] Ущерб: +{check_b['objective'] - check['objective']} у.е.")
            scenarios.append({'Scenario': 'Arc Failure', 'Objective': check_b['objective'], 'Status': 'Optimal'})
        else:
            scenarios.append({'Scenario': 'Arc Failure', 'Objective': None, 'Status': 'Infeasible'})

        # Рост спроса
        data_loader.generate_and_save_data(seed, add_trips=1)
        drivers_g, trips_g, costs_g = data_loader.load_data(n_trips=7)
        res_g, _, check_g = solve_and_verify(drivers_g, trips_g, costs_g)

        if res_g.success:
            scenarios.append({'Scenario': 'Demand +16%', 'Objective': check_g['objective'], 'Status': 'Optimal'})
        else:
            print("[Сценарий Рост] Водителей не хватает (Infeasible).")
            scenarios.append({'Scenario': 'Demand +16%', 'Objective': None, 'Status': 'Infeasible'})

        pd.DataFrame(scenarios).to_csv("results/scenarios.csv", index=False)
        print("\n=== Все эксперименты успешно завершены! ===")

    except Exception:
        print("\n!!! ПРОИЗОШЛА ОШИБКА !!!")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10109)
    args = parser.parse_args()
    main(args.seed)