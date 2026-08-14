"""SQLModel-таблица рыночных событий раунда.

Событие — внешний шок, меняющий рынок между раундами независимо от решений
команд (см. :mod:`core.market_events`). Хранится отдельной строкой на раунд,
а не полем в :class:`~db.models.Round`, по трём причинам:

* событий за раунд может быть несколько, и они перемножаются;
* у каждого есть свой текст для витрины (заголовок + описание);
* базовые ``market_a``/``market_b``/``market_mc`` раунда остаются нетронутыми —
  всегда видно, каким рынок был до события и каким стал после.

Существующие таблицы (:mod:`db.models`, :mod:`db.role_models`) не меняются.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel

from db.enums import EventKind

__all__ = ["MarketEvent"]


def _utcnow() -> datetime:
    """Вернуть текущий момент в UTC (timezone-aware, без deprecated utcnow)."""
    return datetime.now(UTC)


class MarketEvent(SQLModel, table=True):
    """Одно рыночное событие раунда: относительный шок + текст для витрины.

    Уникальности по (round, kind) нет намеренно: два шока спроса за раунд —
    допустимый сценарий, они перемножаются (``combine_shocks``).
    """

    id: int | None = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="round.id", index=True)
    kind: EventKind
    # Относительное изменение параметра: +0.10 = +10%. Строго больше -1;
    # проверку выполняет core.market_events.MarketShock при применении.
    magnitude: float
    headline: str
    description: str = ""
    # Видно ли событие командам ДО подачи решения. True — опубликованный шок
    # (команды учитывают его в расчёте); False — сюрприз, витрина покажет его
    # только после закрытия раунда. На расчёт рынка флаг не влияет.
    revealed: bool = True
    # Ключ пресета из core.market_events.EVENT_PRESETS, если событие взято из
    # библиотеки; пустая строка — событие заведено профессором вручную.
    preset_key: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
