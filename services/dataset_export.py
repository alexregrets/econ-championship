"""Комната данных: то, что команда скачивает и анализирует.

Из этого файла студент получает всё, что знает о рынке. Отсюда два правила,
которые важнее любых удобств.

**Истина наружу не уходит.** Ни настоящих ``a`` и ``b``, ни пофирменных
издержек из :class:`~db.role_models.CompanyGroundTruth`. Выдать их числом —
значит отменить смысл раунда: оценивать станет нечего. Ошибка коварна тем, что
снаружи выгрузка выглядит нормально, поэтому на неё стоит отдельный тест.

**Названия компаний настоящие, цифры выдуманные.** Это стандартная практика
бизнес-школ (disguised case), но она обязана быть подписана — см.
:data:`DISCLAIMER` и ``CASES.md``. Без подписи это фальшивая отчётность
реальной компании; с подписью — учебный кейс.

Про формат
----------
CSV отдаётся **чистым**: только шапка и данные, без комментариев в начале
файла. Комментарии удобны для pandas, но студенты этого курса открывают
выгрузку Excel и Gretl, а те на строках с ``#`` спотыкаются. Словарь данных
живёт отдельно: вторым листом XLSX и отдельным markdown-файлом.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from openpyxl import Workbook
from sqlmodel.ext.asyncio.session import AsyncSession

from core.cases import build_case
from core.dataset import HistorySpec
from core.market_engine import MarketParameters, nash_equilibrium
from db import repositories as repo
from db.enums import Method

__all__ = [
    "DISCLAIMER",
    "FORBIDDEN_COLUMNS",
    "Column",
    "Dataset",
    "build_round_dataset",
    "to_csv",
    "to_xlsx",
    "data_dictionary_markdown",
]

DISCLAIMER = (
    "Учебные данные. Смоделированы для турнира, отчётностью компаний не являются."
)

# Столбцы, которых в выгрузке быть не может ни под каким видом: это и есть та
# истина, ради оценки которой раунд существует. Проверяется тестом, а не
# бдительностью — бдительность кончается на третьем месяце разработки.
FORBIDDEN_COLUMNS = frozenset(
    {
        "market_a",
        "market_b",
        "market_mc",
        "a",
        "b",
        "marginal_cost",
        "implied_marginal_cost",
        "ground_truth",
        "true_price",
        "nash",
    }
)

# Сколько наблюдений видит команда. Тридцать — компромисс: меньше двадцати
# оценка наклона шумит сильнее эффекта, больше сотни превращает разбор в
# работу с большими данными, которой курс не занимается.
DEFAULT_PERIODS = 30

_METHOD_TITLES: dict[Method, str] = {
    Method.OLS_SIMPLE: "Парная регрессия спроса",
    Method.OLS_MULTIPLE: "Множественная регрессия и фиктивные переменные",
    Method.MULTICOLLINEARITY: "Мультиколлинеарность",
    Method.HETEROSCEDASTICITY: "Гетероскедастичность",
    Method.AUTOCORRELATION: "Автокорреляция",
    Method.PANEL_DATA: "Панельные данные",
}


# Доля от предельно допустимого разброса выпуска. Единица оставила бы цену
# ровно на уровне себестоимости в худшем периоде — без запаса под шум.
_SPREAD_SAFETY = 0.7

# Шум как доля полосы между равновесной ценой и себестоимостью. Пятнадцать
# процентов оставляют сигнал заметно сильнее шума на тридцати наблюдениях.
_NOISE_SHARE = 0.15


def _scaled_noise_and_spread(
    params: MarketParameters, n_firms: int
) -> dict[str, float]:
    """Подобрать шум и разброс выпуска под масштаб конкретного рынка.

    Задавать их абсолютными числами нельзя, и это выяснилось дорого. Шум в 20
    единиц цены — пустяк для нефтяного кейса, где цена около 785 $/т, и больше
    всей полосы между ценой и себестоимостью для учебного рынка, где
    равновесная цена 21. Одна и та же константа в одном случае незаметна, в
    другом делает данные нечитаемыми.

    Границы считаются из самого рынка:

    * Равновесный суммарный выпуск ``Q* = n(a−c)/((n+1)b)`` составляет ``n/(n+1)``
      от выпуска ``(a−c)/b``, при котором цена падает до себестоимости. Значит
      разброс выпуска больше ``1/n`` увёл бы цену ниже издержек — история, в
      которой фирмы устойчиво работают в убыток, выглядит нелепо и портит
      оценку. Берём ``_SPREAD_SAFETY`` от этого предела.
    * Шум привязан к полосе ``P* − c``: она и есть весь экономический размах
      рынка, и мерить отклонения имеет смысл в её долях.
    """
    nash_price = (params.a + n_firms * params.marginal_cost) / (n_firms + 1)
    price_band = nash_price - params.marginal_cost

    return {
        "quantity_spread": _SPREAD_SAFETY / n_firms,
        "noise_sd": max(_NOISE_SHARE * price_band, 0.0),
    }


@dataclass(frozen=True)
class Column:
    """Столбец выгрузки вместе с объяснением, что это такое.

    Столбец без единиц и описания бесполезен: студент не может ни
    проинтерпретировать коэффициент, ни понять, что с чем сравнивать.
    Поэтому описание — обязательное поле, а не необязательное украшение.
    """

    name: str
    unit: str
    description: str


@dataclass(frozen=True)
class Dataset:
    """Готовая к выдаче таблица наблюдений раунда."""

    round_number: int
    method: Method
    title: str
    columns: tuple[Column, ...]
    rows: list[dict[str, float]]

    @property
    def filename_stem(self) -> str:
        return f"round_{self.round_number:02d}_{self.method.value}"


async def build_round_dataset(session: AsyncSession, round_id: int) -> Dataset:
    """Собрать рыночную историю раунда для выдачи командам.

    Зерном генерации служит сам ``round_id``. Это не заглушка: раунд —
    естественный ключ данных, зерно получается стабильным без единого нового
    столбца в схеме, а значит без миграции. Когда Alembic встанет (см. гейт в
    ``LOOP_PROMPT.md``), зерно можно будет вынести в поле ``Round`` и задавать
    вручную; до тех пор менять схему ради этого дороже, чем польза.

    Raises
    ------
    ValueError
        Если раунда не существует.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")

    params = MarketParameters(
        a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
    )

    # История строится вокруг равновесного выпуска рынка: наблюдения должны
    # лежать в том же диапазоне, в котором команда потом принимает решение.
    # Иначе она экстраполирует регрессию за пределы данных и не узнает об этом.
    teams = await repo.list_teams(session)
    n_firms = max(len(teams), 1)
    reference_quantity = nash_equilibrium(n_firms, params) * n_firms

    # Данные строятся под метод раунда: в раунде по фиктивным переменным
    # в них лежит смена режима, в раунде по парной регрессии — чистый спрос.
    # Метод без своего кейса роняет выгрузку намеренно (см. core.cases):
    # выдать вместо него baseline значило бы объявить раунд по методу,
    # которому в данных нечего искать.
    case = build_case(
        round_.method,
        params,
        reference_quantity=reference_quantity,
        spec=HistorySpec(
            periods=DEFAULT_PERIODS,
            seed=round_id,
            **_scaled_noise_and_spread(params, n_firms),
        ),
    )

    columns = (
        Column("period", "номер", "Порядковый номер периода наблюдения"),
        Column(
            "quantity",
            "млн т",
            "Суммарный выпуск рынка за период — сумма по всем фирмам",
        ),
        Column("price", "$/т", "Рыночная цена, сложившаяся при этом выпуске"),
        *(Column(c.name, c.unit, c.description) for c in case.extra_columns),
    )

    forbidden = {c.name for c in columns} & FORBIDDEN_COLUMNS
    if forbidden:
        raise ValueError(
            f"кейс метода {round_.method.value} объявил запрещённые столбцы: "
            f"{sorted(forbidden)}. Через выгрузку истина наружу не уходит"
        )

    rows = [
        {
            "period": float(o.period),
            "quantity": o.quantity,
            "price": o.price,
            **{name: float(value) for name, value in o.extras.items()},
        }
        for o in case.observations
    ]

    return Dataset(
        round_number=round_.number,
        method=round_.method,
        title=_METHOD_TITLES.get(round_.method, round_.method.value),
        columns=columns,
        rows=rows,
    )


def to_csv(dataset: Dataset) -> str:
    """CSV без комментариев: шапка и данные, больше ничего."""
    buffer = io.StringIO()
    names = [c.name for c in dataset.columns]
    writer = csv.DictWriter(buffer, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    for row in dataset.rows:
        writer.writerow(row)
    return buffer.getvalue()


def data_dictionary_markdown(dataset: Dataset) -> str:
    """Словарь данных — отдельным файлом, рядом с CSV."""
    lines = [
        f"# Раунд {dataset.round_number} — {dataset.title}",
        "",
        f"> {DISCLAIMER}",
        "",
        "## Переменные",
        "",
        "| Столбец | Единицы | Что это |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| `{c.name}` | {c.unit} | {c.description} |" for c in dataset.columns
    )
    lines.extend(
        [
            "",
            f"Наблюдений: {len(dataset.rows)}.",
            "",
            "Параметры рынка в выгрузку не входят намеренно — их и предстоит "
            "оценить по этим данным.",
        ]
    )
    return "\n".join(lines)


def to_xlsx(dataset: Dataset) -> bytes:
    """XLSX двумя листами: «Данные» и «Словарь»."""
    book = Workbook()

    sheet = book.active
    assert sheet is not None
    sheet.title = "Данные"
    sheet.append([c.name for c in dataset.columns])
    for row in dataset.rows:
        sheet.append([row[c.name] for c in dataset.columns])

    dictionary = book.create_sheet("Словарь")
    dictionary.append([f"Раунд {dataset.round_number} — {dataset.title}"])
    dictionary.append([DISCLAIMER])
    dictionary.append([])
    dictionary.append(["Столбец", "Единицы", "Что это"])
    for column in dataset.columns:
        dictionary.append([column.name, column.unit, column.description])
    dictionary.append([])
    dictionary.append(
        [
            "Параметры рынка в выгрузку не входят намеренно — "
            "их и предстоит оценить по этим данным."
        ]
    )

    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()
