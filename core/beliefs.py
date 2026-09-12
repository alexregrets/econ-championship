"""Обратный ход: по сданному объёму — во что команда верила про рынок.

Раунд закрыт, цены и прибыли посчитаны. Этот модуль отвечает на другой вопрос:
**какой рынок команда держала в голове, когда выбирала объём.** Ответ
превращает разбор из «вы проиграли» в «вы действовали так, будто ждали цену
640; рынок дал 512; разница — цена незамеченного сдвига».

Механизм. Команда ``i`` решала задачу лучшего ответа на остаточный спрос
``P = A_i - b * q_i``, где ``A_i = a - b * Q_{-i}`` — цена насыщения за вычетом
фактического выпуска соперников. Из условия первого порядка
``A_i - c_i - 2 * b_hat * q_i = 0`` вынимаются три величины разбора:

* **неявный наклон** ``b_hat = (A_i - c_i) / (2 * q_i)`` — какой крутизны
  спрос делает выбранный объём оптимальным. Меньше истинного ``b`` — команда
  сочла спрос более пологим, чем он есть, и перепроизвела.
* **ожидаемая цена** ``P_ожид = c_i + b * q_i`` — цена, к которой команда
  вышла бы, читая истинную крутизну (формула из ``CASES.md``). Разрыв с
  фактической ценой — то, чего команда не увидела в данных раунда.
* **разрыв в прибыли** — насколько меньше команда заработала против лучшего
  ответа на тот же выпуск соперников.

Неявный наклон и ожидаемая цена — два независимых среза, а не одна модель:
первый смотрит на крутизну через остаточный спрос, вторая — на уровень через
истинный ``b``. В истинном равновесии Нэша оба среза сходятся и оба разрыва
равны нулю; расходятся они ровно там, где команда ошиблась.

Как и весь ``core``: чистые функции, только стандартная библиотека, никакого
I/O и обращений к БД. Истинные ``a``, ``b`` и издержки приходят параметром —
это вход разбора у преподавателя, студентам он недоступен.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from core.market_engine import MarketParameters, compute_cournot_round

__all__ = ["RecoveredBelief", "recover_beliefs"]


@dataclass(frozen=True)
class RecoveredBelief:
    """Восстановленные убеждения одной команды по итогам раунда.

    Всё, кроме ``team_id`` и ``submitted_quantity``, — производные величины
    разбора. Знаки выбраны так, чтобы «плохо» читалось как «положительное»:
    ``price_gap > 0`` — команда ждала цену выше рынка, ``profit_gap > 0`` —
    недобрала прибыли против лучшего ответа.

    Attributes
    ----------
    team_id:
        Идентификатор команды.
    submitted_quantity:
        Объём, который команда сдала за раунд.
    rivals_quantity:
        Суммарный фактический выпуск всех остальных команд, ``Q_{-i}``.
    residual_intercept:
        ``A_i = a - b * Q_{-i}`` — цена насыщения остаточного спроса, с которым
        команда фактически столкнулась.
    implied_slope:
        ``b_hat`` — наклон спроса, при котором сданный объём оптимален.
    expected_price:
        ``c_i + b * q_i`` — цена, на которую команда вышла бы при истинном
        наклоне (формула ``CASES.md``).
    actual_price:
        Фактическая цена закрытия рынка (одна на всех, из движка раунда).
    price_gap:
        ``expected_price - actual_price``.
    best_response_quantity:
        Объём лучшего ответа на тот же выпуск соперников при истинных
        параметрах, ``(A_i - c_i) / (2 * b)``; ноль, если остаточный спрос уже
        ниже издержек.
    profit:
        Фактическая прибыль команды за раунд (из движка).
    best_response_profit:
        Прибыль, которую команда получила бы, сыграв ``best_response_quantity``
        против неизменного выпуска соперников.
    profit_gap:
        ``best_response_profit - profit``; неотрицателен по определению
        лучшего ответа.
    """

    team_id: str
    submitted_quantity: float
    rivals_quantity: float
    residual_intercept: float
    implied_slope: float
    expected_price: float
    actual_price: float
    price_gap: float
    best_response_quantity: float
    profit: float
    best_response_profit: float
    profit_gap: float


def recover_beliefs(
    decisions: Mapping[str, float],
    params: MarketParameters,
    *,
    marginal_costs: Mapping[str, float] | None = None,
) -> dict[str, RecoveredBelief]:
    """Восстановить убеждения каждой команды по сданным объёмам.

    Parameters
    ----------
    decisions:
        ``team_id -> объём`` за раунд. Тот же вход, что у
        :func:`core.market_engine.compute_cournot_round`. Объёмы конечны и
        строго положительны: из нулевого объёма убеждение не восстановить —
        команда, которая ничего не произвела, ничего и не сообщила о рынке.
    params:
        Истинные параметры рынка раунда. Вход разбора, не выгрузки студентам.
    marginal_costs:
        Пофирменные издержки ``team_id -> c_i`` для асимметричного раунда.
        ``None`` — все фирмы на общей ``params.marginal_cost``.

    Returns
    -------
    dict[str, RecoveredBelief]
        По одной записи на команду, ключ — ``team_id``.

    Raises
    ------
    ValueError
        Если ``decisions`` пуст, объём неположителен либо не конечен, или
        ``marginal_costs`` заданы не для всех команд.
    """
    if not decisions:
        raise ValueError("decisions must contain at least one team")

    for team_id, quantity in decisions.items():
        if not math.isfinite(quantity):
            raise ValueError(f"team {team_id!r} has non-finite quantity {quantity}")
        if quantity <= 0:
            raise ValueError(
                f"team {team_id!r} produced a non-positive quantity ({quantity}): "
                "belief cannot be recovered from no action"
            )

    costs = _resolve_costs(decisions, params, marginal_costs)

    # Фактическая цена — из того же движка, что считает раунд: разбор обязан
    # говорить о тех же числах, которые видели команды.
    results = compute_cournot_round(decisions, params)
    total_quantity = math.fsum(decisions.values())

    beliefs: dict[str, RecoveredBelief] = {}
    for team_id, quantity in decisions.items():
        cost = costs[team_id]
        rivals_quantity = total_quantity - quantity
        residual_intercept = params.a - params.b * rivals_quantity

        implied_slope = (residual_intercept - cost) / (2.0 * quantity)
        expected_price = cost + params.b * quantity

        best_response_quantity = (residual_intercept - cost) / (2.0 * params.b)
        if best_response_quantity <= 0.0:
            # Остаточный спрос ниже издержек: любой положительный объём убыточен,
            # лучший ответ — не производить вовсе.
            best_response_quantity = 0.0
            best_response_profit = 0.0
        else:
            best_response_profit = params.b * best_response_quantity**2

        actual_price = results[team_id].price
        # Прибыль — по издержкам самой команды, а не по общей издержке рынка:
        # симметричный движок знает только params.marginal_cost, а в
        # асимметричном раунде у каждой фирмы свой c_i.
        profit = (actual_price - cost) * quantity

        beliefs[team_id] = RecoveredBelief(
            team_id=team_id,
            submitted_quantity=quantity,
            rivals_quantity=rivals_quantity,
            residual_intercept=residual_intercept,
            implied_slope=implied_slope,
            expected_price=expected_price,
            actual_price=actual_price,
            price_gap=expected_price - actual_price,
            best_response_quantity=best_response_quantity,
            profit=profit,
            best_response_profit=best_response_profit,
            profit_gap=best_response_profit - profit,
        )
    return beliefs


def _resolve_costs(
    decisions: Mapping[str, float],
    params: MarketParameters,
    marginal_costs: Mapping[str, float] | None,
) -> dict[str, float]:
    """Развернуть издержки до ``team_id -> c_i``, проверив полноту набора."""
    if marginal_costs is None:
        return {team_id: params.marginal_cost for team_id in decisions}

    missing = set(decisions) - set(marginal_costs)
    if missing:
        raise ValueError(
            f"marginal_costs заданы не для всех команд, нет: {sorted(missing)}"
        )
    return {team_id: float(marginal_costs[team_id]) for team_id in decisions}
