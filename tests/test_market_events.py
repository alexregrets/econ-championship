"""Тесты чистого движка рыночных событий (core/market_events.py).

Ключевые гарантии:
- NEUTRAL: пустой набор событий не меняет ни один параметр рынка — раунды без
  событий считаются ровно как до появления модуля;
- порядок событий не влияет на результат (мультипликативная композиция);
- каждый вид шока трогает ровно свой параметр и не задевает остальные;
- рынок, сломанный событиями (издержки догнали спрос), отвергается с явной
  ошибкой, а не считается молча;
- величина шока валидируется (конечна, строго больше −1).
"""

from __future__ import annotations

import math

import pytest

from core.market_engine import MarketParameters, compute_cournot_round
from core.market_events import (
    EVENT_PRESETS,
    MarketShock,
    apply_to_costs,
    apply_to_demand,
    apply_to_parameters,
    combine_shocks,
    preset_by_key,
    shock_summary,
)
from db.enums import EventKind


@pytest.fixture
def params() -> MarketParameters:
    """Тот же generic-рынок, что в остальных тестах движков."""
    return MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)


# --------------------------------------------------------------------------- #
# Нейтральность и композиция
# --------------------------------------------------------------------------- #


def test_no_events_leaves_market_untouched(params: MarketParameters) -> None:
    """NEUTRAL: без событий параметры совпадают побитово."""
    shocked = apply_to_parameters(params, [])

    assert shocked.a == params.a
    assert shocked.b == params.b
    assert shocked.marginal_cost == params.marginal_cost


def test_no_events_gives_identical_round(params: MarketParameters) -> None:
    """REGRESSION: результат раунда без событий совпадает с прежним движком."""
    decisions = {"1": 20.0, "2": 25.0, "3": 30.0}

    baseline = compute_cournot_round(decisions, params)
    with_events = compute_cournot_round(decisions, apply_to_parameters(params, []))

    assert with_events == baseline


def test_combine_is_order_independent() -> None:
    """Два шока в любом порядке дают одни и те же множители."""
    demand = MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.25)
    cost = MarketShock(kind=EventKind.COST_SHOCK, magnitude=0.18)

    forward = combine_shocks([demand, cost])
    backward = combine_shocks([cost, demand])

    assert forward == backward


def test_same_kind_shocks_multiply() -> None:
    """Два шока спроса −10% и −20% дают 0.72, а не 0.70."""
    factors = combine_shocks(
        [
            MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.10),
            MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.20),
        ]
    )

    assert factors.demand == pytest.approx(0.72)
    assert factors.slope == 1.0
    assert factors.cost == 1.0


# --------------------------------------------------------------------------- #
# Каждый вид шока трогает свой параметр
# --------------------------------------------------------------------------- #


def test_demand_shift_moves_only_intercept(params: MarketParameters) -> None:
    shocked = apply_to_parameters(
        params, [MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.25)]
    )

    assert shocked.a == pytest.approx(75.0)
    assert shocked.b == params.b
    assert shocked.marginal_cost == params.marginal_cost


def test_elasticity_shift_moves_only_slope(params: MarketParameters) -> None:
    shocked = apply_to_parameters(
        params, [MarketShock(kind=EventKind.ELASTICITY_SHIFT, magnitude=0.15)]
    )

    assert shocked.a == params.a
    assert shocked.b == pytest.approx(1.15)
    assert shocked.marginal_cost == params.marginal_cost


def test_cost_shock_moves_only_marginal_cost(params: MarketParameters) -> None:
    shocked = apply_to_parameters(
        params, [MarketShock(kind=EventKind.COST_SHOCK, magnitude=0.18)]
    )

    assert shocked.a == params.a
    assert shocked.b == params.b
    assert shocked.marginal_cost == pytest.approx(11.8)


def test_apply_to_demand_matches_parameters(params: MarketParameters) -> None:
    """Спросовая ветка (асимметричный раунд) согласована с симметричной."""
    shocks = [
        MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.12),
        MarketShock(kind=EventKind.ELASTICITY_SHIFT, magnitude=0.15),
    ]

    a, b = apply_to_demand(params.a, params.b, shocks)
    symmetric = apply_to_parameters(params, shocks)

    assert a == pytest.approx(symmetric.a)
    assert b == pytest.approx(symmetric.b)


def test_demand_shock_does_not_touch_per_firm_costs() -> None:
    """Шок спроса не двигает c_i — только COST_SHOCK."""
    costs = {"1": 55.0, "2": 472.0}

    shocked = apply_to_costs(
        costs,
        [MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.25)],
        demand_intercept=2000.0,
    )

    assert shocked == costs


def test_cost_shock_scales_every_firm() -> None:
    """COST_SHOCK применяется к каждой фирме одним множителем."""
    costs = {"1": 100.0, "2": 200.0}

    shocked = apply_to_costs(
        costs,
        [MarketShock(kind=EventKind.COST_SHOCK, magnitude=0.10)],
        demand_intercept=2000.0,
    )

    assert shocked == pytest.approx({"1": 110.0, "2": 220.0})


# --------------------------------------------------------------------------- #
# Отказ считать сломанный рынок
# --------------------------------------------------------------------------- #


def test_events_breaking_symmetric_market_raise(params: MarketParameters) -> None:
    """Издержки выше точки насыщения — ошибка с упоминанием событий."""
    with pytest.raises(ValueError, match="нежизнеспособ"):
        apply_to_parameters(
            params,
            [
                MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.95),
                MarketShock(kind=EventKind.COST_SHOCK, magnitude=1.0),
            ],
        )


def test_events_breaking_asymmetric_market_name_the_teams() -> None:
    """В ошибке перечислены команды, чья себестоимость догнала спрос."""
    with pytest.raises(ValueError, match=r"\['2'\]"):
        apply_to_costs(
            {"1": 50.0, "2": 900.0},
            [MarketShock(kind=EventKind.COST_SHOCK, magnitude=0.20)],
            demand_intercept=1000.0,
        )


def test_magnitude_must_be_above_minus_one() -> None:
    with pytest.raises(ValueError, match="больше -1"):
        MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-1.0)


def test_magnitude_must_be_finite() -> None:
    with pytest.raises(ValueError, match="конечной"):
        MarketShock(kind=EventKind.COST_SHOCK, magnitude=math.inf)


# --------------------------------------------------------------------------- #
# Сводка и библиотека пресетов
# --------------------------------------------------------------------------- #


def test_summary_is_empty_without_events() -> None:
    assert shock_summary([]) == ""


def test_summary_mentions_every_touched_parameter() -> None:
    summary = shock_summary(
        [
            MarketShock(kind=EventKind.DEMAND_SHIFT, magnitude=-0.12),
            MarketShock(kind=EventKind.COST_SHOCK, magnitude=0.18),
        ]
    )

    assert "спрос -12.0%" in summary
    assert "издержки +18.0%" in summary


def test_presets_have_unique_keys() -> None:
    keys = [preset.key for preset in EVENT_PRESETS]

    assert len(keys) == len(set(keys))


def test_every_preset_converts_to_a_valid_shock() -> None:
    """Ни один пресет библиотеки не нарушает валидацию величины."""
    for preset in EVENT_PRESETS:
        shock = preset.to_shock()

        assert shock.kind is preset.kind
        assert shock.headline == preset.headline


def test_presets_keep_the_oil_market_viable() -> None:
    """Любой одиночный пресет оставляет калиброванный нефтяной рынок живым."""
    from devshell.role_seed import oil_2013_market_parameters

    base = oil_2013_market_parameters()
    for preset in EVENT_PRESETS:
        shocked = apply_to_parameters(base, [preset.to_shock()])

        assert shocked.marginal_cost < shocked.a


def test_unknown_preset_key_raises() -> None:
    with pytest.raises(KeyError):
        preset_by_key("не-существует")
