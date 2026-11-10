# lab1_inventory.py
import io
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-режим: графики только сохраняются
import matplotlib.pyplot as plt
from scipy.optimize import linprog

SEED = 10109
PRODUCTS = [f"P{i}" for i in range(1, 6)]
WEEKS = [1, 2, 3, 4]
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

DATA_ITEMS = "items_weeks.csv"
DATA_RES = "resources.csv"


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def generate_data(seed=SEED):
    """Генерирует данные с фиксированным seed.

    Особенности:
      1) holding_cost задаётся ОДИН РАЗ на товар (а не на пару товар-неделя);
      2) purchase_limit = 1.2 * max недельного спроса, чтобы в базовом
         сценарии узким местом был СКЛАД, а не лимит закупки.
         Тогда дефицитный сценарий (cap_mult=0.8) реально меняет план,
         а стрессовый (dem_mult=1.2, pur_mult=0.8) остаётся infeasible.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in PRODUCTS:
        # ОДИН раз на товар — стоимость хранения не зависит от недели
        holding_cost = float(1 + rng.integers(0, 2))          # 1..2
        for t in WEEKS:
            demand = float(10 + rng.integers(-2, 3))          # 8..12
            if t == 1:
                purchase_cost = float(4 + rng.integers(0, 2))  # 4..5
            else:
                purchase_cost = float(10 + rng.integers(0, 2)) # 10..11
            rows.append({
                "product": p,
                "week": t,
                "demand": demand,
                "purchase_cost": purchase_cost,
                "holding_cost": holding_cost,
                "volume": 1.0,
            })
    df = pd.DataFrame(rows)

    weekly_demand = df.groupby("week")["demand"].sum()
    smax = float(weekly_demand.max())
    res = pd.DataFrame({
        "week": WEEKS,
        "capacity": [10.0] * len(WEEKS),
        "purchase_limit": [round(1.2 * smax, 2)] * len(WEEKS),
    })

    df.to_csv(DATA_ITEMS, index=False)
    res.to_csv(DATA_RES, index=False)
    return df, res


def load_data():
    """Читает данные из CSV. Задействована в __main__ (требование п. 1.2.3
    и п. 1.2.5: данные отдельно от модели, отдельная функция загрузки)."""
    return pd.read_csv(DATA_ITEMS), pd.read_csv(DATA_RES)


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
def build_model(df, res, cap_mult=1.0, dem_mult=1.0, pur_mult=1.0):
    nP, nT = len(PRODUCTS), len(WEEKS)
    n = 2 * nP * nT

    ix = {(p, t): i * nT + (t - 1)
          for i, p in enumerate(PRODUCTS) for t in WEEKS}
    iI = {(p, t): nP * nT + i * nT + (t - 1)
          for i, p in enumerate(PRODUCTS) for t in WEEKS}

    c = np.zeros(n)
    d = {}
    for _, r in df.iterrows():
        p, t = r["product"], int(r["week"])
        d[(p, t)] = r["demand"] * dem_mult
        c[ix[(p, t)]] = r["purchase_cost"]
        c[iI[(p, t)]] = r["holding_cost"]

    Aeq, beq = [], []
    for p in PRODUCTS:
        for t in WEEKS:
            row = np.zeros(n)
            row[iI[(p, t)]] = 1.0
            if t > 1:
                row[iI[(p, t - 1)]] -= 1.0
            row[ix[(p, t)]] -= 1.0
            Aeq.append(row)
            beq.append(-d[(p, t)])

    Aub, bub = [], []
    for _, rr in res.iterrows():
        t = int(rr["week"])
        row = np.zeros(n)
        for p in PRODUCTS:
            vol = df.loc[
                (df["product"] == p) & (df["week"] == t), "volume"
            ].iloc[0]
            row[iI[(p, t)]] = vol
        Aub.append(row)
        bub.append(rr["capacity"] * cap_mult)

    for _, rr in res.iterrows():
        t = int(rr["week"])
        row = np.zeros(n)
        for p in PRODUCTS:
            row[ix[(p, t)]] = 1.0
        Aub.append(row)
        bub.append(rr["purchase_limit"] * pur_mult)

    return dict(
        c=c,
        Aub=np.array(Aub), bub=np.array(bub),
        Aeq=np.array(Aeq), beq=np.array(beq),
        bounds=[(0, None)] * n,
        ix=ix, iI=iI,
    )


# ---------------------------------------------------------------------------
# Решение
# ---------------------------------------------------------------------------
def solve_model(m):
    """Запускает HiGHS через scipy.optimize.linprog.
    Весь stdout солвера перехватывается и дописывается в solver.log."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = linprog(
            m["c"],
            A_ub=m["Aub"], b_ub=m["bub"],
            A_eq=m["Aeq"], b_eq=m["beq"],
            bounds=m["bounds"],
            method="highs",
            options={"disp": True},
        )
    with open("solver.log", "a", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return res


def solution_table(m, res):
    """Вектор res.x → таблица (product, week, purchase, inventory_end).
    Вызывать только при res.success == True."""
    rows = []
    for p in PRODUCTS:
        for t in WEEKS:
            rows.append({
                "product": p,
                "week": t,
                "purchase": res.x[m["ix"][(p, t)]],
                "inventory_end": res.x[m["iI"][(p, t)]],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Независимая проверка (NumPy, без объекта солвера)
# ---------------------------------------------------------------------------
def independent_check(df, res, sol, cap_mult=1.0, dem_mult=1.0, pur_mult=1.0):
    prev = {p: 0.0 for p in PRODUCTS}
    max_bal = max_cap = max_pur = 0.0
    min_nonneg = 0.0
    obj = 0.0

    for t in WEEKS:
        totI = totX = 0.0
        for p in PRODUCTS:
            r = df.loc[(df["product"] == p) & (df["week"] == t)].iloc[0]
            s = sol[(p, t)]
            x, I = s["purchase"], s["inventory_end"]
            min_nonneg = min(min_nonneg, x, I)
            max_bal = max(max_bal,
                          abs(I - prev[p] - x + r["demand"] * dem_mult))
            obj += r["purchase_cost"] * x + r["holding_cost"] * I
            totI += r["volume"] * I
            totX += x
            prev[p] = I

        cap = res.loc[res["week"] == t, "capacity"].iloc[0] * cap_mult
        plim = res.loc[res["week"] == t, "purchase_limit"].iloc[0] * pur_mult
        max_cap = max(max_cap, totI - cap)
        max_pur = max(max_pur, totX - plim)

    return dict(
        max_balance_violation=max_bal,
        max_capacity_violation=max_cap,
        max_purchase_violation=max_pur,
        min_nonneg_violation=min_nonneg,
        objective_recomputed=float(obj),
    )


# ---------------------------------------------------------------------------
# Анализ infeasible
# ---------------------------------------------------------------------------
def analyze_infeasibility(df, res, dem_mult=1.0, pur_mult=1.0):
    """Кумулятивный анализ: задача неразрешима, если хотя бы на одном
    префиксе недель суммарный спрос превышает суммарный лимит закупки
    (при нулевом начальном запасе). Это соответствует детекции
    infeasible в presolve HiGHS."""
    rows, cum_d, cum_l = [], 0.0, 0.0
    for t in WEEKS:
        d_t = float(df.loc[df["week"] == t, "demand"].sum()) * dem_mult
        l_t = (float(res.loc[res["week"] == t, "purchase_limit"].iloc[0])
               * pur_mult)
        cum_d += d_t
        cum_l += l_t
        rows.append({
            "week": t,
            "weekly_demand": round(d_t, 4),
            "weekly_limit": round(l_t, 4),
            "cum_demand": round(cum_d, 4),
            "cum_limit": round(cum_l, 4),
            "cum_deficit": round(cum_d - cum_l, 4),
            "conflict": bool(cum_d - cum_l > 1e-9),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
def plot_scenario(st, name, out_dir=PLOTS_DIR):
    """Столбчатые диаграммы закупки и запаса по неделям (стек по товарам)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    pivot_x = st.pivot(index="week", columns="product", values="purchase")
    pivot_I = st.pivot(index="week", columns="product", values="inventory_end")

    pivot_x.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10")
    axes[0].set_title(f"Закупка по неделям (сценарий: {name})")
    axes[0].set_xlabel("Неделя")
    axes[0].set_ylabel("ед.")
    axes[0].legend(title="Товар", loc="upper right", fontsize=8)

    pivot_I.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
    axes[1].set_title(f"Запас на конец недели (сценарий: {name})")
    axes[1].set_xlabel("Неделя")
    axes[1].set_ylabel("ед.")
    axes[1].legend(title="Товар", loc="upper right", fontsize=8)

    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_scenarios_comparison(summary, out_dir=PLOTS_DIR):
    """Сравнение ЦФ по сценариям. infeasible-сценарий подсвечен красным."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(summary.index)
    values = [v if v is not None else 0 for v in summary["objective"]]
    colors = ["steelblue" if s == "optimal" else "salmon"
              for s in summary["status"]]
    ax.bar(names, values, color=colors)
    top = max(values) if values else 1.0
    for i, (_, v, s) in enumerate(zip(names, values, summary["status"])):
        label = f"{s}\n{v:.1f}" if v else s
        ax.text(i, (v if v else 0) + top * 0.02, label,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Целевая функция, руб.")
    ax.set_title("Сравнение сценариев")
    fig.tight_layout()
    path = out_dir / "scenarios_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Один сценарий
# ---------------------------------------------------------------------------
def run_scenario(name, df, res, cap_mult=1.0, dem_mult=1.0, pur_mult=1.0):
    m = build_model(df, res, cap_mult, dem_mult, pur_mult)
    r = solve_model(m)

    status = {
        0: "optimal", 1: "iteration limit", 2: "infeasible",
        3: "unbounded", 4: "numerical",
    }.get(r.status, str(r.status))

    print(f"\n=== Сценарий '{name}': status={status}, fun={r.fun} ===")

    info = {
        "name": name,
        "status": status,
        "objective": float(r.fun) if r.success else None,
    }

    if r.success:
        st = solution_table(m, r)
        st.to_csv(f"solution_{name}.csv", index=False)

        sol = {(row["product"], int(row["week"])): {
                   "purchase": row["purchase"],
                   "inventory_end": row["inventory_end"]}
               for _, row in st.iterrows()}

        chk = independent_check(df, res, sol, cap_mult, dem_mult, pur_mult)
        print("Независимая проверка:", chk)

        # --- явная проверка с допуском 1e-7 (требование п. 1.1.2) ---
        tol = 1e-7
        assert chk["max_balance_violation"] <= tol, \
            f"[{name}] нарушен баланс запаса: {chk['max_balance_violation']}"
        assert chk["max_capacity_violation"] <= tol, \
            f"[{name}] превышена вместимость склада: {chk['max_capacity_violation']}"
        assert chk["max_purchase_violation"] <= tol, \
            f"[{name}] превышен лимит закупки: {chk['max_purchase_violation']}"
        assert chk["min_nonneg_violation"] >= -tol, \
            f"[{name}] отрицательная переменная: {chk['min_nonneg_violation']}"
        assert abs(chk["objective_recomputed"] - r.fun) <= tol, \
            (f"[{name}] ЦФ не совпадает: пересчёт "
             f"{chk['objective_recomputed']} ≠ солвер {r.fun}")
        print(f"Проверка с допуском {tol:g}: PASSED")

        agg = st.groupby("week").agg(
            purchase=("purchase", "sum"),
            inventory_end=("inventory_end", "sum"),
        ).reset_index()
        print(agg.to_string(index=False))

        print("Теневые цены (capacity):", r.ineqlin.marginals[:4])
        print("Теневые цены (purchase):", r.ineqlin.marginals[4:8])

        path = plot_scenario(st, name)
        print(f"График сохранён: {path}")
    else:
        conflict = analyze_infeasibility(df, res, dem_mult, pur_mult)
        conflict.to_csv(f"conflict_{name}.csv", index=False)
        print("Анализ конфликтов ограничений:")
        print(conflict.to_string(index=False))
        bad = conflict[conflict["conflict"]]["week"].tolist()
        if bad:
            print(f"КОНФЛИКТ: недели {bad}: спрос превышает лимит закупки, "
                  f"при нулевом начальном запасе задача неразрешима.")

    return info


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Данные — отдельно от модели: генерируем один раз, затем читаем из CSV.
    # Это одновременно закрывает требования п. 1.2.3 и п. 1.2.5.
    if not (Path(DATA_ITEMS).exists() and Path(DATA_RES).exists()):
        print(f"Генерация данных (seed={SEED})…")
        generate_data()
    df, res = load_data()

    print(f"\nВариант V=7, seed={SEED}")
    print("\nСгенерированные данные:")
    print(df.to_string(index=False))
    print("\nРесурсы по неделям:")
    print(res.to_string(index=False))

    results = [
        run_scenario("base", df, res),
        run_scenario("deficit", df, res, cap_mult=0.8),
        run_scenario("stress", df, res, dem_mult=1.2, pur_mult=0.8),
    ]

    summary = pd.DataFrame([
        {"scenario": r["name"],
         "status": r["status"],
         "objective": r["objective"]}
        for r in results
    ]).set_index("scenario")
    summary.to_csv("scenarios_summary.csv")
    print("\n=== Сводная таблица сценариев ===")
    print(summary.to_string())

    cmp_path = plot_scenarios_comparison(summary)
    print(f"Сравнительный график: {cmp_path}")

    # --- минимальный regression-тест сценариев ---
    z_base = results[0]["objective"]
    z_deficit = results[1]["objective"]
    assert results[0]["status"] == "optimal"
    assert results[1]["status"] == "optimal"
    assert results[2]["status"] == "infeasible"
    assert z_deficit > z_base + 1e-9

    print("\nВсе контрольные проверки пройдены:")
    print(f"  base    = {z_base:.2f} руб.")
    print(f"  deficit = {z_deficit:.2f} руб. (рост на {z_deficit - z_base:.2f} руб.)")