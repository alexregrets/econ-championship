"""Детектор «кто попался»: неявный наклон команды против наивной оценки.

Защита турнира от «скинул всё задание в ИИ» строится как payoff-дизайн, не
как детектор текста (решение 08.09). Этот модуль — вторая её половина: после
раунда неявный наклон ``b̂`` команды (из :mod:`core.beliefs`) сравнивается
с тем, что даёт наивная сквозная регрессия на тех же данных. Совпало —
флаг преподавателю: команда действовала по смещённой оценке, метод раунда
не применяла.

Ловушка есть только там, где метод смещает **точечную** оценку. Парная
регрессия и гетероскедастичность оценку не смещают — там детектору нечего
ловить, и он честно возвращает :attr:`Verdict.NO_TRAP`, а не выдумывает
вердикт по шуму. Правило выведено 27.08 при калибровке кейсов.

Как и весь ``core``: чистые функции, стандартная библиотека, без I/O.
"""

from __future__ import annotations

import math
from enum import StrEnum

from core.cases import DEFAULT_REGIME_SHIFT
from db.enums import Method

__all__ = ["DEFAULT_TOLERANCE", "Verdict", "naive_slope_ratio", "trap_verdict"]

# Относительный допуск в долях истинного наклона. Полосы «верно» и
# «попался» при 0.35 не пересекаются: [0.8, 1.2] против [0.15, 0.55].
DEFAULT_TOLERANCE = 0.2


class Verdict(StrEnum):
    """Что говорит неявный наклон команды о её процедуре."""

    SOUND = "sound"  # близко к истинному b — метод применён
    TRAPPED = "trapped"  # близко к наивной оценке — ловушка сработала
    OFF = "off"  # ни то ни другое — ошибка иного рода, смотреть отчёт
    NO_TRAP = "no_trap"  # метод раунда точечную оценку не смещает


# Доля истинного наклона, которую видит наивная процедура по методу раунда.
# ``None`` — наивная оценка не смещена, деньгами такой раунд не судится.
_NAIVE_RATIOS: dict[Method, float | None] = {
    Method.OLS_SIMPLE: None,
    Method.OLS_MULTIPLE: DEFAULT_REGIME_SHIFT.naive_slope_ratio,
    Method.HETEROSCEDASTICITY: None,
}


def naive_slope_ratio(method: Method) -> float | None:
    """``b_наивный / b`` для метода раунда; ``None`` — ловушки нет.

    Метод без кейса тоже даёт ``None``: разбор закрытого раунда не должен
    падать из-за того, что детектор про метод ничего не знает.
    """
    return _NAIVE_RATIOS.get(method)


def trap_verdict(
    implied_slope: float,
    true_slope: float,
    naive_ratio: float | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Verdict:
    """Отнести неявный наклон команды к одной из полос.

    Parameters
    ----------
    implied_slope:
        ``b̂`` команды из :func:`core.beliefs.recover_beliefs`.
    true_slope:
        Истинный ``b`` рынка раунда (после событий).
    naive_ratio:
        :func:`naive_slope_ratio` метода раунда; ``None`` → :attr:`Verdict.NO_TRAP`.
    tolerance:
        Полуширина полосы в долях ``true_slope``.

    Raises
    ------
    ValueError
        Наклоны неположительны или не конечны; полосы «верно» и «попался»
        пересекаются при заданном допуске.
    """
    if not math.isfinite(implied_slope) or implied_slope <= 0.0:
        raise ValueError(f"implied_slope must be finite and positive, got {implied_slope}")
    if not math.isfinite(true_slope) or true_slope <= 0.0:
        raise ValueError(f"true_slope must be finite and positive, got {true_slope}")
    if naive_ratio is None:
        return Verdict.NO_TRAP
    if not 0.0 < naive_ratio < 1.0:
        raise ValueError(f"naive_ratio must lie in (0, 1), got {naive_ratio}")
    if tolerance <= 0.0 or 2.0 * tolerance >= 1.0 - naive_ratio:
        raise ValueError(
            f"tolerance {tolerance} makes the 'sound' and 'trapped' bands overlap "
            f"for naive_ratio {naive_ratio}; it must be below {(1.0 - naive_ratio) / 2:.3f}"
        )

    ratio = implied_slope / true_slope
    if abs(ratio - 1.0) <= tolerance:
        return Verdict.SOUND
    if abs(ratio - naive_ratio) <= tolerance:
        return Verdict.TRAPPED
    return Verdict.OFF
