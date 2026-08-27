"""Кейсы раундов — данные, в которые заложена закономерность метода раунда.

Модуль отвечает на один вопрос: **чем данные раунда по методу X отличаются от
данных раунда по методу Y**. До него отличий не было — все шесть методов курса
получали одну и ту же baseline-историю из :mod:`core.dataset`, и раунд по
фиктивным переменным ничем не отличался от раунда по парной регрессии.

Это и есть тот самый молчаливый провал, о котором предупреждает
``LOOP_PROMPT.md``: студент честно применяет метод, ничего не находит, и вина
выглядит его. Поэтому здесь действует правило: **метод без своего кейса падает
с** :class:`NotImplementedError`, а не подменяется baseline. Громкий отказ на
настройке раунда стоит минуты, молчаливая подмена — всего турнира.

Кейсы строятся по ``CASES.md``, а не придумываются заново. Данные фабрикуются
(решение от 14 августа): названия компаний настоящие, цифры смоделированы,
поэтому каждая выгрузка несёт дисклеймер из :mod:`services.dataset_export`.

Как и весь ``core``: только стандартная библиотека, никакого I/O и никаких
обращений к БД. Научный стек живёт в тестах — именно они проверяют, что
заложенная закономерность действительно лежит в данных.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from core.dataset import MIN_PERIODS, HistorySpec, Observation, generate_market_history
from core.market_engine import MarketParameters
from db.enums import Method

__all__ = [
    "REGIME_COLUMN",
    "ColumnSpec",
    "CaseData",
    "RegimeShiftSpec",
    "DEFAULT_REGIME_SHIFT",
    "build_case",
    "supported_methods",
]

# Имя режимного столбца — одно на весь проект. Строкой его больше нигде не
# писать: выгрузка, тесты и словарь данных обязаны спотыкаться вместе, а не
# расходиться по опечатке.
REGIME_COLUMN = "regime_new"


@dataclass(frozen=True)
class ColumnSpec:
    """Описание дополнительного столбца кейса.

    Единицы и описание обязательны: столбец без них студент не может
    проинтерпретировать, а значит и включить в регрессию осмысленно.
    """

    name: str
    unit: str
    description: str


@dataclass(frozen=True)
class CaseData:
    """Готовые наблюдения раунда вместе с описанием своих столбцов.

    Attributes
    ----------
    observations:
        История наблюдений, периоды пронумерованы сквозняком с единицы.
    extra_columns:
        Столбцы сверх ``period``/``quantity``/``price``. У baseline пусто.
    """

    observations: list[Observation]
    extra_columns: tuple[ColumnSpec, ...] = ()


@dataclass(frozen=True)
class RegimeShiftSpec:
    """Настройка режимного сдвига — раунд 2, «Смена покупателей».

    Сюжет. Раньше сырьё уходило на глубокий рынок с множеством покупателей:
    дополнительная партия почти не двигала цену, остаточный спрос был
    эластичным, а объёмы — скромнее. После переориентации экспорта круг
    покупателей сузился: продаётся больше, но **каждая дополнительная партия
    идёт с растущим дисконтом** — спрос стал жёстким. Рынок сменил режим,
    а не просто подвинулся.

    Attributes
    ----------
    historic_slope_ratio:
        ``b_исторический / b_текущий``. Меньше единицы — раньше спрос был
        эластичнее. Эта разница и делает нужным взаимодействие дамми
        с объёмом: без неё режим сводился бы к сдвигу константы.
    naive_slope_ratio:
        Доля от истинного наклона, которую увидит наивная сквозная регрессия.
        **Это и есть ручка силы ловушки**, и она выставляется прямо, а не
        получается сама собой — см. :func:`_build_regime_shift`.
    quantity_gap_ratio:
        Насколько ниже был исторический выпуск, в долях от коридора
        ``quantity_spread``. Привязка к коридору, а не к абсолютной величине:
        коридор сам считается от числа фирм, и кластеры обязаны разъезжаться
        вместе с ним, иначе калибровка едет при смене состава турнира.
    shift_share:
        Доля наблюдений, приходящаяся на исторический режим.

    Raises
    ------
    ValueError
        Если пропорции выходят за допустимые границы.
    """

    historic_slope_ratio: float = 0.5
    naive_slope_ratio: float = 0.35
    quantity_gap_ratio: float = 2.0
    shift_share: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.historic_slope_ratio < 1.0:
            raise ValueError(
                "historic_slope_ratio должен лежать в (0, 1): исторический спрос "
                f"эластичнее текущего, получено {self.historic_slope_ratio}"
            )
        if not 0.0 < self.naive_slope_ratio < 1.0:
            raise ValueError(
                "naive_slope_ratio должен лежать в (0, 1): ловушка обязана "
                "занижать наклон. Завышение наклона в Курно почти ничего "
                f"не стоит, см. docstring _build_regime_shift; получено "
                f"{self.naive_slope_ratio}"
            )
        if self.quantity_gap_ratio <= 0.0:
            raise ValueError(
                "quantity_gap_ratio должен быть положительным: без разъезда "
                f"кластеров сквозная регрессия не смещается, получено "
                f"{self.quantity_gap_ratio}"
            )
        if not 0.0 < self.shift_share < 1.0:
            raise ValueError(
                "shift_share должен лежать в (0, 1): нужны оба режима, "
                f"получено {self.shift_share}"
            )


DEFAULT_REGIME_SHIFT = RegimeShiftSpec()


def _regime_seed(seed: int, index: int) -> int:
    """Зерно отдельного режима.

    Режимы генерируются двумя вызовами, и одинаковое зерно дало бы им
    одинаковый рисунок отклонений — совпадение, которое студент заметил бы
    раньше, чем закономерность.
    """
    return seed * 2 + index


def _level_shift_for_naive_slope(
    old: list[Observation],
    new: list[Observation],
    *,
    target_slope: float,
) -> float:
    """Подобрать сдвиг уровня, при котором сквозной МНК даёт заданный наклон.

    Сквозной наклон равен ``S_qp / S_qq``. Прибавление константы ``δ`` к ценам
    одного из режимов не трогает ``S_qq`` и меняет ``S_qp`` ровно на ``δ·G``,
    где ``G = Σ(Q_i − Q̄)`` по сдвигаемому режиму. Отсюда ``δ`` выражается
    в одну строку — подбирать его перебором не нужно.

    ``G`` обращается в ноль, только если центр сдвигаемого режима совпал
    с общим центром: тогда уровнем на сквозной наклон повлиять нельзя.

    Parameters
    ----------
    old, new:
        Наблюдения двух режимов до сдвига уровня.
    target_slope:
        Желаемый ``b`` наивной сквозной регрессии для модели ``P = a − b·Q``.

    Raises
    ------
    ValueError
        Если режимы наблюдаются в одном и том же диапазоне выпуска.
    """
    everything = old + new
    total = len(everything)
    mean_q = sum(o.quantity for o in everything) / total
    mean_p = sum(o.price for o in everything) / total

    s_qq = sum((o.quantity - mean_q) ** 2 for o in everything)
    s_qp = sum((o.quantity - mean_q) * (o.price - mean_p) for o in everything)
    gap = sum(o.quantity - mean_q for o in old)

    if abs(gap) < 1e-9:
        raise ValueError(
            "режимы наблюдаются вокруг одного и того же выпуска: сдвигом уровня "
            "сквозной наклон не задать. Увеличьте quantity_gap_ratio"
        )

    # Спрос убывает, поэтому целевому b соответствует наклон регрессии −b.
    return (-target_slope * s_qq - s_qp) / gap


def _renumber(observations: list[Observation], start: int, regime: float) -> list[Observation]:
    """Перенумеровать периоды сквозняком и проставить флаг режима."""
    return [
        replace(o, period=start + offset, extras={REGIME_COLUMN: regime})
        for offset, o in enumerate(observations)
    ]


def _build_simple(
    params: MarketParameters,
    *,
    reference_quantity: float,
    spec: HistorySpec,
) -> CaseData:
    """Раунд 1 — baseline без ловушек. Задача: научиться читать спрос по точкам."""
    history = generate_market_history(
        params, reference_quantity=reference_quantity, spec=spec
    )
    return CaseData(observations=history)


def _build_regime_shift(
    params: MarketParameters,
    *,
    reference_quantity: float,
    spec: HistorySpec,
    shift: RegimeShiftSpec = DEFAULT_REGIME_SHIFT,
) -> CaseData:
    """Раунд 2 — два режима спроса в одной выборке.

    ``params`` — параметры **текущего** режима, они же истина раунда: именно
    в нём команда принимает решение. Исторический режим строится вокруг
    меньшего выпуска и более пологой кривой.

    Почему ловушка обязана **занижать** наклон
    ------------------------------------------
    В равновесии Курно плечо экстраполяции жёстко связано с числом фирм:
    расстояние от наблюдаемого выпуска до выпуска, при котором цена падает
    до себестоимости, составляет ``Q/n``. Поэтому ошибка в наклоне бьёт
    несимметрично. Если команда ставит долю ``k`` от правильного объёма,
    а соперники играют равновесно, её прибыль равна ``k(2−k)`` от правильной,
    то есть теряется ``(1−k)²``. Завышенный наклон загоняет ``k`` в потолок
    ``n/(n+1)`` — при четырёх фирмах это потеря максимум 4 % и никакого урока.
    Заниженный наклон не ограничен ничем: ``b_hat = 0.3·b`` при четырёх фирмах
    стоит уже 22 % прибыли. **Дорого ошибиться можно только в одну сторону**,
    и кейс обязан вести именно туда. Числа проверены отдельным прогоном
    и записаны в ``LOOP_LOG.md``.

    Как задаётся сила ловушки
    -------------------------
    Первая версия оставляла наивный наклон на волю случая: у режимов были
    разные опорные выпуски, сквозная регрессия ловила межкластерный наклон,
    и цена ошибки гуляла **между 12 % и 200 % прибыли в зависимости от зерна**.
    Раунд выигрывался бы жребием, а не анализом — ровно то, из-за чего
    14 августа отказались от реальной калибровки (``CASES.md``).

    Здесь наивный наклон выставляется **точно**. Уровень исторического режима
    сдвигается на константу ``δ``, подобранную по уже сгенерированным точкам
    так, чтобы сквозной МНК дал в точности ``naive_slope_ratio · b``. Сдвиг
    уровня не трогает ни внутренние наклоны режимов, ни разброс выпуска,
    поэтому правильная спецификация с дамми по-прежнему видит правду,
    а неправильная — заданную ложь. Одна ручка, воспроизводимо на любом зерне.

    Raises
    ------
    ValueError
        Если на один из режимов приходится меньше :data:`core.dataset.MIN_PERIODS`
        наблюдений, либо подобранный уровень уводит историческую цену
        ниже себестоимости — такую историю нельзя показывать студентам.
    """
    periods_old = round(spec.periods * shift.shift_share)
    periods_new = spec.periods - periods_old
    if min(periods_old, periods_new) < MIN_PERIODS:
        raise ValueError(
            f"на режим приходится {min(periods_old, periods_new)} наблюдений, "
            f"а для отдельного наклона нужно минимум {MIN_PERIODS}: увеличьте "
            f"periods (сейчас {spec.periods}) или сдвиньте shift_share"
        )

    # Исторический рынок: выпуск ниже, спрос эластичнее. Уровень пока
    # нейтральный — цена в его центре равна текущей; настоящий уровень
    # подбирается ниже, когда точки уже есть.
    historic_reference = reference_quantity * (
        1.0 - shift.quantity_gap_ratio * spec.quantity_spread
    )
    if historic_reference <= 0:
        raise ValueError(
            f"исторический выпуск ушёл в {historic_reference:.1f}: "
            f"quantity_gap_ratio={shift.quantity_gap_ratio} слишком велик "
            f"для коридора {spec.quantity_spread:.2f}"
        )
    historic_slope = params.b * shift.historic_slope_ratio
    current_price = params.a - params.b * reference_quantity
    historic = MarketParameters(
        a=current_price + historic_slope * historic_reference,
        b=historic_slope,
        marginal_cost=params.marginal_cost,
    )

    old = generate_market_history(
        historic,
        reference_quantity=historic_reference,
        spec=replace(spec, periods=periods_old, seed=_regime_seed(spec.seed, 0)),
    )
    new = generate_market_history(
        params,
        reference_quantity=reference_quantity,
        spec=replace(spec, periods=periods_new, seed=_regime_seed(spec.seed, 1)),
    )

    level_shift = _level_shift_for_naive_slope(
        old, new, target_slope=params.b * shift.naive_slope_ratio
    )
    shifted_old = [replace(o, price=o.price + level_shift) for o in old]

    floor = min(o.price for o in shifted_old)
    if floor <= params.marginal_cost:
        raise ValueError(
            f"подобранный уровень исторического режима опускает цену до "
            f"{floor:.1f} при себестоимости {params.marginal_cost:.1f}: история, "
            f"в которой отрасль устойчиво работает в убыток, не годится "
            f"в учебные данные. Ослабьте naive_slope_ratio "
            f"({shift.naive_slope_ratio}) или quantity_gap_ratio "
            f"({shift.quantity_gap_ratio})"
        )

    observations = _renumber(shifted_old, start=1, regime=0.0) + _renumber(
        new, start=periods_old + 1, regime=1.0
    )
    columns = (
        ColumnSpec(
            REGIME_COLUMN,
            "0/1",
            "Режим рынка: 0 — до смены режима, 1 — после. Момент смены известен",
        ),
    )
    return CaseData(observations=observations, extra_columns=columns)


# Тип строителя кейса. Именованно, чтобы реестр читался, а не расшифровывался.
CaseBuilder = Callable[..., CaseData]

_BUILDERS: dict[Method, CaseBuilder] = {
    Method.OLS_SIMPLE: _build_simple,
    Method.OLS_MULTIPLE: _build_regime_shift,
}


def supported_methods() -> tuple[Method, ...]:
    """Методы, под которые кейс уже написан.

    Остальные существуют в ``CASES.md`` как сценарии, но данных под них нет,
    и раунд по ним настроить нельзя — см. :func:`build_case`.
    """
    return tuple(_BUILDERS)


def build_case(
    method: Method,
    params: MarketParameters,
    *,
    reference_quantity: float,
    spec: HistorySpec,
) -> CaseData:
    """Построить данные раунда под его метод.

    Parameters
    ----------
    method:
        Метод раунда. Он определяет, какая закономерность лежит в данных.
    params:
        Истинные параметры **текущего** рынка. Командам не показываются.
    reference_quantity:
        Опорный выпуск, вокруг которого гуляет история текущего режима.
    spec:
        Общие настройки генерации: сколько периодов, зерно, шум, разброс.

    Raises
    ------
    NotImplementedError
        Если под метод ещё нет кейса. Осознанно громко: молчаливая выдача
        baseline означала бы раунд, в котором заявленный метод нечего
        обнаруживать, и заметили бы это последним.
    """
    builder = _BUILDERS.get(method)
    if builder is None:
        raise NotImplementedError(
            f"под метод {method.value} кейс ещё не написан. Сценарий есть "
            f"в CASES.md, данных нет. Реализованы: "
            f"{', '.join(m.value for m in supported_methods())}"
        )
    return builder(params, reference_quantity=reference_quantity, spec=spec)
