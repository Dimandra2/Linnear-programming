# ЛР №1. ЛП-модель управления запасами склада

Вариант V=7: 5 товаров, 4 недели.
Seed генерации: 10109.

## Установка
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Запуск одной командой
python main.py

## Что появится после запуска
- items_weeks.csv, resources.csv — сгенерированные данные;
- solver.log — лог HiGHS по всем сценариям;
- solution_base.csv, solution_deficit.csv — решения;
- conflict_stress.csv — анализ конфликта для infeasible-сценария;
- scenarios_summary.csv — ЦФ и статусы по сценариям;
- plots/base.png, plots/deficit.png, plots/scenarios_comparison.png — графики.

## Модель
min Σ(c·x + h·I)
s.t. I_{i,t} = I_{i,t-1} + x_{i,t} - d_{i,t}
     Σ v_i I_{i,t} ≤ C_t
     Σ x_{i,t} ≤ P_t
     x, I ≥ 0

## Сценарии
- base: без изменений;
- deficit: вместимость склада ×0.8;
- stress: спрос ×1.2, лимит закупки ×0.8 → infeasible (анализируется в conflict_stress.csv).