"""Генератор рыночной истории раунда.

Это спецификация, написанная до кода. Проверяется не «функция что-то вернула»,
а два свойства, без которых датасет бесполезен:

1. **Воспроизводимость.** Одно зерно — один и тот же датасет, всегда. Без этого
   невозможен разбор после раунда: препод не сможет показать те же числа,
   которые видела команда.
2. **Оценимость.** По сгенерированным данным МНК обязан восстанавливать
   параметры спроса, которые в них заложены. Датасет, из которого не выводится
   правда, — это молчаливый провал: студент честно применит метод, получит
   мусор и не поймёт, чья это вина.

Второе свойство проверяется настоящим `statsmodels`, а не пересчётом формулы
руками: если наша арифметика и его разойдутся, знать об этом надо здесь, а не
на турнире.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from core.dataset import HistorySpec, Observation, generate_market_history
from core.market_engine import MarketParameters

PARAMS = MarketParameters(a=2000.0, b=3.6, marginal_cost=364.0)
REFERENCE_Q = 350.0


def _fit(observations: list[Observation]) -> tuple[float, float]:
    """Вернуть (a_hat, b_hat) из МНК-регрессии цены на объём."""
    quantities = np.array([o.quantity for o in observations])
    prices = np.array([o.price for o in observations])
    model = sm.OLS(prices, sm.add_constant(quantities)).fit()
    intercept, slope = model.params
    return float(intercept), float(-slope)  # спрос убывает: P = a - b*Q


# --------------------------------------------------------------------------- #
# Воспроизводимость
# --------------------------------------------------------------------------- #


def test_same_seed_gives_identical_history() -> None:
    spec = HistorySpec(periods=30, seed=42)
    first = generate_market_history(PARAMS, reference_quantity=REFERENCE_Q, spec=spec)
    second = generate_market_history(PARAMS, reference_quantity=REFERENCE_Q, spec=spec)
    assert first == second


def test_different_seed_gives_different_history() -> None:
    a = generate_market_history(
        PARAMS, reference_quantity=REFERENCE_Q, spec=HistorySpec(seed=1)
    )
    b = generate_market_history(
        PARAMS, reference_quantity=REFERENCE_Q, spec=HistorySpec(seed=2)
    )
    assert a != b


def test_generation_does_not_touch_global_random() -> None:
    """Свой генератор, а не модуль random: иначе датасет зависит от того,
    что делал процесс до вызова, и воспроизводимость ломается незаметно."""
    import random

    random.seed(0)
    before = random.random()

    random.seed(0)
    generate_market_history(
        PARAMS, reference_quantity=REFERENCE_Q, spec=HistorySpec(seed=7)
    )
    after = random.random()

    assert before == after


# --------------------------------------------------------------------------- #
# Форма
# --------------------------------------------------------------------------- #


def test_period_count_and_numbering() -> None:
    history = generate_market_history(
        PARAMS, reference_quantity=REFERENCE_Q, spec=HistorySpec(periods=24, seed=3)
    )
    assert len(history) == 24
    assert [o.period for o in history] == list(range(1, 25))


def test_prices_and_quantities_stay_positive() -> None:
    history = generate_market_history(
        PARAMS, reference_quantity=REFERENCE_Q, spec=HistorySpec(periods=200, seed=5)
    )
    assert all(o.quantity > 0 for o in history)
    assert all(o.price > 0 for o in history)


def test_quantity_variation_respects_spread() -> None:
    spread = 0.25
    history = generate_market_history(
        PARAMS,
        reference_quantity=REFERENCE_Q,
        spec=HistorySpec(periods=200, seed=11, quantity_spread=spread),
    )
    lowest = min(o.quantity for o in history)
    highest = max(o.quantity for o in history)
    assert lowest >= REFERENCE_Q * (1 - spread) - 1e-9
    assert highest <= REFERENCE_Q * (1 + spread) + 1e-9


def test_too_few_periods_is_rejected() -> None:
    """Меньше десятка точек — регрессия бессмысленна, и лучше сказать это сразу."""
    with pytest.raises(ValueError):
        generate_market_history(
            PARAMS, reference_quantity=REFERENCE_Q, spec=HistorySpec(periods=3)
        )


def test_negative_spread_is_rejected() -> None:
    with pytest.raises(ValueError):
        HistorySpec(quantity_spread=-0.1)


# --------------------------------------------------------------------------- #
# Оценимость — главное свойство
# --------------------------------------------------------------------------- #


def test_zero_noise_lies_exactly_on_the_demand_curve() -> None:
    history = generate_market_history(
        PARAMS,
        reference_quantity=REFERENCE_Q,
        spec=HistorySpec(periods=20, seed=9, noise_sd=0.0),
    )
    for observation in history:
        expected = PARAMS.a - PARAMS.b * observation.quantity
        assert observation.price == pytest.approx(expected)


def test_ols_recovers_planted_parameters() -> None:
    """Baseline первого раунда: студент обязан вытащить a и b обычным МНК."""
    history = generate_market_history(
        PARAMS,
        reference_quantity=REFERENCE_Q,
        spec=HistorySpec(periods=120, seed=13, noise_sd=25.0),
    )
    a_hat, b_hat = _fit(history)

    assert a_hat == pytest.approx(PARAMS.a, rel=0.05)
    assert b_hat == pytest.approx(PARAMS.b, rel=0.05)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_recovery_is_not_a_lucky_seed(seed: int) -> None:
    """Оценимость обязана держаться на любом зерне, а не на удачном."""
    history = generate_market_history(
        PARAMS,
        reference_quantity=REFERENCE_Q,
        spec=HistorySpec(periods=120, seed=seed, noise_sd=25.0),
    )
    _, b_hat = _fit(history)
    assert b_hat == pytest.approx(PARAMS.b, rel=0.10)


def test_noise_is_in_price_units_not_relative() -> None:
    """Больше шума — хуже подгонка. Проверяем, что параметр вообще работает."""
    quiet = generate_market_history(
        PARAMS,
        reference_quantity=REFERENCE_Q,
        spec=HistorySpec(periods=120, seed=17, noise_sd=5.0),
    )
    noisy = generate_market_history(
        PARAMS,
        reference_quantity=REFERENCE_Q,
        spec=HistorySpec(periods=120, seed=17, noise_sd=80.0),
    )
    quiet_error = abs(_fit(quiet)[1] - PARAMS.b)
    noisy_error = abs(_fit(noisy)[1] - PARAMS.b)
    assert noisy_error > quiet_error
