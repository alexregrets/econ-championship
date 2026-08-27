"""Данные под метод раунда — спецификация, написанная до кода.

Это самый важный контракт всего турнира. У каждого раунда есть свой
``Round.method``, и данные этого раунда обязаны содержать ровно ту
закономерность, которую метод должен обнаружить. Если не содержат — студент
честно применяет метод, ничего не находит, и вся конструкция превращается
в театр: эконометрика перестаёт быть путём к решению и становится ритуалом.

Поэтому проверяется не «функция вернула список», а три свойства:

1. **Закономерность действительно заложена.** Проверяется настоящим
   ``statsmodels``, а не пересчётом формулы руками.
2. **Правильный метод восстанавливает правду.** Спецификация с дамми и
   взаимодействием обязана вернуть оба наклона близко к истинным.
3. **Наивный метод даёт заметно смещённый ответ, и это стоит денег.**
   Если ошибка ничего не стоит, у команды нет причины стараться.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from core.cases import (
    DEFAULT_REGIME_SHIFT,
    REGIME_COLUMN,
    CaseData,
    build_case,
    supported_methods,
)
from core.dataset import HistorySpec, generate_market_history
from core.market_engine import MarketParameters
from db.enums import Method
from services.dataset_export import DEFAULT_PERIODS, _scaled_noise_and_spread

# Текущий режим рынка — он же истина раунда: команда принимает решение здесь.
PARAMS = MarketParameters(a=2000.0, b=3.6, marginal_cost=364.0)
REFERENCE_Q = 350.0
SPEC = HistorySpec(periods=40, seed=7, noise_sd=12.0, quantity_spread=0.18)
N_FIRMS = 4


def _arrays(case: CaseData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вернуть (Q, P, regime) numpy-массивами."""
    quantities = np.array([o.quantity for o in case.observations])
    prices = np.array([o.price for o in case.observations])
    regime = np.array([o.extras[REGIME_COLUMN] for o in case.observations])
    return quantities, prices, regime


def _naive_slope(case: CaseData) -> tuple[float, float]:
    """Парная регрессия по всей выборке — то, что сделает наивная команда.

    Возвращает (a_hat, b_hat) для модели ``P = a - b*Q``.
    """
    quantities, prices, _ = _arrays(case)
    fit = sm.OLS(prices, sm.add_constant(quantities)).fit()
    return float(fit.params[0]), float(-fit.params[1])


def _dummy_slopes(case: CaseData) -> tuple[float, float]:
    """Регрессия с дамми режима и взаимодействием — правильный путь.

    Возвращает (b_старого_режима, b_текущего_режима).
    """
    quantities, prices, regime = _arrays(case)
    design = np.column_stack([quantities, regime, regime * quantities])
    fit = sm.OLS(prices, sm.add_constant(design)).fit()
    slope_old = -float(fit.params[1])
    slope_new = -float(fit.params[1] + fit.params[3])
    return slope_old, slope_new


def _cournot_quantity(a: float, b: float, cost: float, n_firms: int) -> float:
    """Объём одной фирмы в симметричном равновесии Курно при её вере в (a, b)."""
    return (a - cost) / ((n_firms + 1) * b)


def _profit(own_q: float, rivals_q: float, params: MarketParameters) -> float:
    """Прибыль фирмы при её объёме и суммарном объёме соперников."""
    price = params.a - params.b * (own_q + rivals_q)
    return (price - params.marginal_cost) * own_q


def _market_for(n_firms: int, seed: int) -> tuple[MarketParameters, float, HistorySpec]:
    """Собрать раунд ровно так, как его собирает боевая выгрузка.

    Шум и коридор выпуска берутся не из головы, а из той же функции, которой
    пользуется :mod:`services.dataset_export`. Иначе калибровка проверялась бы
    на настройках, которых на турнире не бывает.
    """
    reference_quantity = (
        n_firms * (PARAMS.a - PARAMS.marginal_cost) / ((n_firms + 1) * PARAMS.b)
    )
    spec = HistorySpec(
        periods=DEFAULT_PERIODS,
        seed=seed,
        **_scaled_noise_and_spread(PARAMS, n_firms),
    )
    return PARAMS, reference_quantity, spec


# --------------------------------------------------------------------------- #
# Каркас: реестр и baseline
# --------------------------------------------------------------------------- #


def test_supported_methods_are_registered() -> None:
    """Реестр не пуст и содержит только значения Method."""
    methods = supported_methods()
    assert methods
    assert all(isinstance(m, Method) for m in methods)


def test_unsupported_method_fails_loudly() -> None:
    """Метод без кейса обязан падать, а не молча отдавать baseline.

    Молчаливая подмена — это ровно тот провал, который заметен последним:
    раунд по гетероскедастичности прошёл бы на данных без гетероскедастичности,
    и никто бы не понял, почему тесты студентов ничего не отвергают.
    """
    missing = next((m for m in Method if m not in supported_methods()), None)
    if missing is None:
        pytest.skip("все шесть методов уже реализованы")
    with pytest.raises(NotImplementedError):
        build_case(missing, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)


def test_simple_regression_case_matches_baseline() -> None:
    """Раунд 1 — тот же baseline, что и раньше, без единого лишнего столбца."""
    case = build_case(Method.OLS_SIMPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)
    baseline = generate_market_history(PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)

    assert case.extra_columns == ()
    assert [o.price for o in case.observations] == [o.price for o in baseline]
    assert all(o.extras == {} for o in case.observations)


# --------------------------------------------------------------------------- #
# Раунд 2 — режимный сдвиг (множественная регрессия и фиктивные переменные)
# --------------------------------------------------------------------------- #


def test_regime_case_is_deterministic() -> None:
    """Одно зерно — один датасет. Без этого разбор после раунда невозможен."""
    first = build_case(Method.OLS_MULTIPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)
    second = build_case(Method.OLS_MULTIPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)
    assert first.observations == second.observations


def test_regime_case_shape() -> None:
    """Периоды сквозные с единицы, режимный столбец объявлен и заполнен."""
    case = build_case(Method.OLS_MULTIPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)

    assert len(case.observations) == SPEC.periods
    assert [o.period for o in case.observations] == list(range(1, SPEC.periods + 1))
    assert [c.name for c in case.extra_columns] == [REGIME_COLUMN]

    flags = [o.extras[REGIME_COLUMN] for o in case.observations]
    assert set(flags) == {0.0, 1.0}
    # Ноль идёт до единицы: режим меняется один раз и не возвращается.
    assert flags == sorted(flags)
    # Каждого режима достаточно, чтобы наклон оценивался по-своему.
    assert flags.count(1.0) >= 10
    assert flags.count(0.0) >= 10


def test_dummy_specification_recovers_current_slope() -> None:
    """Правильный путь возвращает истинный наклон текущего режима.

    Он же наклон раунда: именно по нему команда считает свой объём.

    Допуск 20 %, а не 5 %, и это факт, а не поблажка: на двадцати точках
    с боевым уровнем шума точнее не выходит. Важно другое — наивная
    спецификация мажет втрое, а не на пятую часть.
    """
    case = build_case(Method.OLS_MULTIPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)
    _, slope_new = _dummy_slopes(case)
    assert slope_new == pytest.approx(PARAMS.b, rel=0.20)


def test_regimes_have_genuinely_different_slopes() -> None:
    """Два режима — это два разных наклона, а не один с разным уровнем.

    Иначе взаимодействие дамми с объёмом не нужно, и весь метод раунда
    сводится к сдвигу константы. Текущий режим жёстче исторического:
    круг покупателей сузился, и дополнительный объём давит цену сильнее.
    """
    case = build_case(Method.OLS_MULTIPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)
    slope_old, slope_new = _dummy_slopes(case)
    assert slope_new > slope_old * 1.3


def test_naive_pooled_regression_gets_exactly_the_planted_slope() -> None:
    """Наивный наклон — заданное число, а не то, что получилось.

    Сила ловушки — единственная ручка кейса, и она обязана выдерживаться
    на любом зерне. Иначе раунд выигрывает тот, кому повезло с данными:
    ровно та беда, из-за которой отказались от реальной калибровки.
    """
    case = build_case(Method.OLS_MULTIPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)
    _, naive_b = _naive_slope(case)
    planted = PARAMS.b * DEFAULT_REGIME_SHIFT.naive_slope_ratio
    assert naive_b == pytest.approx(planted, rel=1e-6)


def test_naive_estimate_costs_real_money() -> None:
    """Ошибка обязана быть экономически ощутимой, иначе стараться незачем.

    Команда с наивной оценкой видит пологий спрос, перепроизводит,
    обваливает цену и теряет прибыль против команды, которая заметила
    смену режима.
    """
    case = build_case(Method.OLS_MULTIPLE, PARAMS, reference_quantity=REFERENCE_Q, spec=SPEC)
    naive_a, naive_b = _naive_slope(case)

    correct_q = _cournot_quantity(PARAMS.a, PARAMS.b, PARAMS.marginal_cost, N_FIRMS)
    naive_q = _cournot_quantity(naive_a, naive_b, PARAMS.marginal_cost, N_FIRMS)

    # Наивный ставит больше: пологий спрос обещает, что цена почти не упадёт.
    assert naive_q > correct_q

    # Соперники играют равновесно, разница — только в собственном объёме.
    rivals = correct_q * (N_FIRMS - 1)
    correct_profit = _profit(correct_q, rivals, PARAMS)
    naive_profit = _profit(naive_q, rivals, PARAMS)

    assert naive_profit < correct_profit
    loss_share = (correct_profit - naive_profit) / correct_profit
    assert loss_share > 0.08


@pytest.mark.parametrize("n_firms", [3, 4, 5, 6])
def test_trap_stays_inside_calibration_corridor(n_firms: int) -> None:
    """Цена ошибки держится в коридоре при любом составе турнира.

    Это и есть ответ на открытый вопрос №1 из ``STATE.md``: слишком мягко —
    ловушку никто не заметит, слишком грубо — заметят все, и разбора
    не выйдет. Коридор проверяется на четырёх составах и восьми зёрнах,
    а не на одном удачном прогоне.

    Нижняя граница — 3 %: столько стоит ошибка, которую студент спишет
    на шум. Верхняя — 45 %: дороже уже не урок, а вылет из турнира.
    """
    losses = []
    for seed in range(1, 9):
        params, reference_q, spec = _market_for(n_firms, seed)
        case = build_case(Method.OLS_MULTIPLE, params, reference_quantity=reference_q, spec=spec)
        naive_a, naive_b = _naive_slope(case)

        correct_q = _cournot_quantity(params.a, params.b, params.marginal_cost, n_firms)
        naive_q = _cournot_quantity(naive_a, naive_b, params.marginal_cost, n_firms)
        rivals = correct_q * (n_firms - 1)

        correct_profit = _profit(correct_q, rivals, params)
        naive_profit = _profit(naive_q, rivals, params)
        losses.append((correct_profit - naive_profit) / correct_profit)

    assert min(losses) > 0.03
    assert max(losses) < 0.45
