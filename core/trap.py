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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from core.cases import DEFAULT_REGIME_SHIFT
from db.enums import Method

__all__ = [
    "DEFAULT_TOLERANCE",
    "ProcedureQuantities",
    "Verdict",
    "naive_slope_ratio",
    "pooled_ols",
    "procedure_quantities",
    "quantity_verdict",
    "trap_verdict",
]

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
    if not math.isfinite(implied_slope):
        raise ValueError(f"implied_slope must be finite, got {implied_slope}")
    if not math.isfinite(true_slope) or true_slope <= 0.0:
        raise ValueError(f"true_slope must be finite and positive, got {true_slope}")
    if naive_ratio is None:
        return Verdict.NO_TRAP
    if implied_slope <= 0.0:
        # Остаточный спрос ушёл под издержки: соперники затопили рынок, и ни
        # один положительный наклон не оправдывает сданный объём. Это не
        # «попался» и не ошибка данных — это OFF, смотреть отчёт команды.
        # Поймано живым прогоном на семи командах, а не тестами.
        return Verdict.OFF
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


# --------------------------------------------------------------------------- #
# Детектор по процедуре: объём против наивного лучшего ответа на тех же данных
# --------------------------------------------------------------------------- #
#
# Почему не по неявному наклону. Живой прогон 12.09: честная наивная команда
# (сквозной МНК по данным раунда) получает наклон ровно 0.35·b, но и точку
# насыщения заниженной — и её объём рационализируется неявным наклоном 0.58·b,
# который в полосу «попался» не попадает. Наклон — не то, что команда подала.
# Подала она объём, и сравнивать надо объёмы: что выдала бы наивная процедура
# на этом датасете при том же ожидании соперников, и что — верная.
#
# Ожидание соперников берётся из истории, а не из фактического раунда:
# (n−1)/n от среднего выпуска текущего режима. Так вердикт зависит от
# процедуры команды, а не от того, что натворили другие в этом раунде — и
# несколько наивных команд не прячут друг друга, как было с наклоном.


@dataclass(frozen=True)
class ProcedureQuantities:
    """Объёмы, которые выдали бы две процедуры на данных раунда."""

    naive_quantity: float  # сквозной МНК → лучший ответ по своей линии
    sound_quantity: float  # истинная линия → лучший ответ
    expected_rivals: float  # общее для обеих ожидание выпуска соперников
    naive_intercept: float
    naive_slope: float


def pooled_ols(rows: Sequence[Mapping[str, float]]) -> tuple[float, float]:
    """Сквозная регрессия цены на выпуск: ``(â, b̂)``, ``b̂ > 0``.

    Ровно то, что делает команда, не заметившая структуры в данных: одна
    линия через всё. Чистый Python — ``core`` не тянет numpy.

    Raises
    ------
    ValueError
        Меньше двух наблюдений или выпуск без разброса.
    """
    n = len(rows)
    if n < 2:
        raise ValueError("pooled OLS needs at least two observations")
    mean_q = math.fsum(r["quantity"] for r in rows) / n
    mean_p = math.fsum(r["price"] for r in rows) / n
    var = math.fsum((r["quantity"] - mean_q) ** 2 for r in rows)
    if var <= 0.0:
        raise ValueError("pooled OLS needs variation in quantity")
    cov = math.fsum((r["quantity"] - mean_q) * (r["price"] - mean_p) for r in rows)
    slope = cov / var
    return mean_p - slope * mean_q, -slope


def procedure_quantities(
    rows: Sequence[Mapping[str, float]],
    *,
    true_a: float,
    true_b: float,
    marginal_cost: float,
    n_firms: int,
    regime_column: str | None = None,
) -> ProcedureQuantities:
    """Что подала бы наивная и что — верная процедура на данных раунда.

    Parameters
    ----------
    rows:
        Наблюдения раунда, как в выгрузке команде: ``quantity``, ``price``,
        при режимном сдвиге — столбец ``regime_column``.
    true_a, true_b, marginal_cost:
        Истина раунда и издержки команды — вход разбора у преподавателя.
    n_firms:
        Фирм на рынке, включая команду.
    regime_column:
        Если задан, ожидание соперников считается по текущему режиму
        (значение столбца ``1``); иначе по всей истории.

    Raises
    ------
    ValueError
        Нет наблюдений текущего режима; ``n_firms < 1``; ``true_b <= 0``.
    """
    if n_firms < 1:
        raise ValueError("n_firms must be at least 1")
    if true_b <= 0.0:
        raise ValueError("true_b must be positive")
    current = (
        [r for r in rows if r.get(regime_column, 1.0) >= 0.5]
        if regime_column is not None
        else list(rows)
    )
    if not current:
        raise ValueError("no observations of the current regime to form expectations")
    recent_total = math.fsum(r["quantity"] for r in current) / len(current)
    expected_rivals = recent_total * (n_firms - 1) / n_firms

    a_hat, b_hat = pooled_ols(rows)
    if b_hat <= 0.0:
        raise ValueError(f"pooled OLS gave a non-negative slope ({-b_hat:g}); data is broken")
    naive = max((a_hat - b_hat * expected_rivals - marginal_cost) / (2.0 * b_hat), 0.0)
    sound = max((true_a - true_b * expected_rivals - marginal_cost) / (2.0 * true_b), 0.0)
    return ProcedureQuantities(
        naive_quantity=naive,
        sound_quantity=sound,
        expected_rivals=expected_rivals,
        naive_intercept=a_hat,
        naive_slope=b_hat,
    )


def quantity_verdict(
    submitted: float,
    procedures: ProcedureQuantities,
    naive_ratio: float | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Verdict:
    """Отнести сданный объём к процедуре, которая его выдала бы.

    ``naive_ratio is None`` — ловушки в методе нет, :attr:`Verdict.NO_TRAP`.
    Иначе объём сравнивается с наивным и верным лучшим ответом в долях
    верного: ближе к верному и в допуске — ``SOUND``; ближе к наивному и в
    допуске — ``TRAPPED``; остальное — ``OFF``. Если процедуры дают почти
    одинаковые объёмы (ловушка на этих данных не сработала), различить их
    нельзя — ``OFF``, а не ложный флаг.

    Raises
    ------
    ValueError
        Объём не конечен или отрицателен; допуск неположителен.
    """
    if not math.isfinite(submitted) or submitted < 0.0:
        raise ValueError(f"submitted quantity must be finite and non-negative, got {submitted}")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if naive_ratio is None:
        return Verdict.NO_TRAP
    sound = procedures.sound_quantity
    naive = procedures.naive_quantity
    if sound <= 0.0:
        return Verdict.OFF
    band = tolerance * sound
    if abs(naive - sound) <= 2.0 * band:
        return Verdict.OFF  # процедуры неразличимы на этих данных
    dist_sound = abs(submitted - sound)
    dist_naive = abs(submitted - naive)
    if dist_sound <= band and dist_sound <= dist_naive:
        return Verdict.SOUND
    if dist_naive <= band and dist_naive < dist_sound:
        return Verdict.TRAPPED
    return Verdict.OFF
