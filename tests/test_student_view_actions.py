"""Тесты read-only действий студенческой витрины и сводки препода.

Тем же стилем, что tests/test_dashboard_actions.py: in-memory база, никакого
Streamlit. Ключевая гарантия витрины — числа предложений ролей скрыты до
фиксации решения lead'ом (не спойлерим внутрикомандное обсуждение).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from dashboard.actions import (
    create_and_open_round,
    latest_round,
    market_brief,
    submit_manual_decision,
    teacher_summary,
    team_role_progress,
)
from db import models  # noqa: F401  (регистрирует таблицы)
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import Role, RoundStatus
from devshell.role_seed import (
    TOTAL_PRODUCTION_2013_MLN_T,
    URALS_PRICE_2013_USD_PER_TON,
    generate_role_views,
    seed_oil_2013,
)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _make_round(session: AsyncSession) -> int:
    round_ = await create_and_open_round(
        session,
        number=1,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="Нефть РФ 2013",
    )
    assert round_.id is not None
    return round_.id


# --------------------------------------------------------------------------- #
# latest_round
# --------------------------------------------------------------------------- #


async def test_latest_round_none_without_rounds(session: AsyncSession) -> None:
    assert await latest_round(session) is None


async def test_latest_round_prefers_open(session: AsyncSession) -> None:
    first = await _make_round(session)
    await repo.set_round_status(session, round_id=first, status=RoundStatus.CLOSED)
    round2 = await repo.create_round(
        session,
        number=2,
        method=(await repo.get_round(session, first)).method,  # type: ignore[union-attr]
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        status=RoundStatus.OPEN,
    )
    found = await latest_round(session)
    assert found is not None
    assert found.id == round2.id


async def test_latest_round_falls_back_to_last_closed(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    await repo.set_round_status(session, round_id=round_id, status=RoundStatus.CLOSED)
    found = await latest_round(session)
    assert found is not None
    assert found.id == round_id
    assert found.status is RoundStatus.CLOSED


# --------------------------------------------------------------------------- #
# market_brief
# --------------------------------------------------------------------------- #


async def test_market_brief_none_without_role_views(session: AsyncSession) -> None:
    round_id = await _make_round(session)
    assert await market_brief(session, round_id) is None


async def test_market_brief_exposes_shared_fields_only(
    session: AsyncSession,
) -> None:
    """Витрина показывает shared buffer срезов: цену и референсный выпуск."""
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)
    brief = await market_brief(session, summary.round_id)
    assert brief is not None
    assert brief.observed_price == pytest.approx(URALS_PRICE_2013_USD_PER_TON)
    assert brief.ref_total_quantity == pytest.approx(TOTAL_PRODUCTION_2013_MLN_T)


# --------------------------------------------------------------------------- #
# team_role_progress: числа скрыты до фиксации lead'ом
# --------------------------------------------------------------------------- #


async def test_progress_hides_numbers_until_lead_locks(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="А", company_name="Роснефть")
    assert team.id is not None
    await role_repo.upsert_role_input(
        session,
        round_id=round_id,
        team_id=team.id,
        role=Role.MARKETER,
        quantity_proposal=120.0,
        note="видел спрос",
    )

    progress = await team_role_progress(session, round_id=round_id, team_id=team.id)

    assert progress.lead_locked is False
    by_role = {row.role: row for row in progress.roles}
    assert len(progress.roles) == 3  # все три роли всегда в списке
    assert by_role[Role.MARKETER].submitted is True
    # Главный инвариант: подача видна, числа — нет.
    assert by_role[Role.MARKETER].quantity is None
    assert by_role[Role.MARKETER].note is None
    assert by_role[Role.FINANCIER].submitted is False


async def test_progress_reveals_numbers_after_lead_locks(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="А", company_name="Роснефть")
    assert team.id is not None
    await role_repo.upsert_role_input(
        session,
        round_id=round_id,
        team_id=team.id,
        role=Role.MARKETER,
        quantity_proposal=120.0,
        note="видел спрос",
    )
    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=110.0, reasoning="lead"
    )

    progress = await team_role_progress(session, round_id=round_id, team_id=team.id)

    assert progress.lead_locked is True
    by_role = {row.role: row for row in progress.roles}
    assert by_role[Role.MARKETER].quantity == pytest.approx(120.0)
    assert by_role[Role.MARKETER].note == "видел спрос"
    # Роль без предложения остаётся пустой и после фиксации.
    assert by_role[Role.FINANCIER].submitted is False
    assert by_role[Role.FINANCIER].quantity is None


# --------------------------------------------------------------------------- #
# teacher_summary
# --------------------------------------------------------------------------- #


async def test_teacher_summary_empty_db(session: AsyncSession) -> None:
    summary = await teacher_summary(session)
    assert summary.teams_total == 0
    assert summary.teams_joined == 0
    assert summary.students_joined == 0
    assert summary.open_round_number is None
    assert summary.decisions_submitted == 0


async def test_teacher_summary_counts_joins_and_decisions(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team_a = await repo.create_team(session, name="А", company_name="Роснефть")
    team_b = await repo.create_team(session, name="Б", company_name="Лукойл")
    assert team_a.id is not None and team_b.id is not None

    # Два студента зашли в команду А, один зарегистрирован, но не привязан.
    joined1 = await repo.create_student(session, telegram_id=1, full_name="s1")
    joined2 = await repo.create_student(session, telegram_id=2, full_name="s2")
    await repo.create_student(session, telegram_id=3, full_name="без команды")
    assert joined1.id is not None and joined2.id is not None
    await repo.assign_student_to_team(
        session, student_id=joined1.id, team_id=team_a.id, role=Role.MARKETER
    )
    await repo.assign_student_to_team(
        session, student_id=joined2.id, team_id=team_a.id, role=Role.FINANCIER
    )

    await submit_manual_decision(
        session, team_id=team_a.id, round_id=round_id, quantity=10.0, reasoning=""
    )

    summary = await teacher_summary(session)
    assert summary.teams_total == 2
    assert summary.teams_joined == 1  # только команда А имеет студентов
    assert summary.students_joined == 2
    assert summary.open_round_number == 1
    assert summary.decisions_submitted == 1


async def test_teacher_summary_no_open_round(session: AsyncSession) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="А", company_name="Роснефть")
    assert team.id is not None
    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=10.0, reasoning=""
    )
    await repo.set_round_status(session, round_id=round_id, status=RoundStatus.CLOSED)

    summary = await teacher_summary(session)
    assert summary.open_round_number is None
    assert summary.decisions_submitted == 0


# --------------------------------------------------------------------------- #
# Комната данных: брифинг + выгрузка для команды
# --------------------------------------------------------------------------- #


async def test_data_room_bundles_brief_and_exports(session: AsyncSession) -> None:
    from dashboard.actions import create_and_open_round, data_room
    from db.enums import Method
    from services.dataset_export import DISCLAIMER

    # Режимный сдвиг откалиброван на 3–6 фирм: коридор выпуска считается от
    # числа команд, и без команд исторический выпуск уходит в минус.
    for i in range(3):
        await repo.create_team(session, name=f"T{i}", company_name=f"C{i}")
    round_ = await create_and_open_round(
        session,
        number=1,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="",
        method=Method.OLS_MULTIPLE,
    )
    assert round_.id is not None
    room = await data_room(session, round_.id)
    assert room is not None

    # Брифинг и словарь несут подпись «учебные данные», CSV — чистый.
    assert DISCLAIMER in room.brief_markdown
    assert DISCLAIMER in room.dictionary_markdown
    assert DISCLAIMER not in room.csv_text
    # В CSV шапка + наблюдения; режимный сдвиг добавляет свой столбец.
    lines = room.csv_text.strip().splitlines()
    assert lines[0].split(",")[:3] == ["period", "quantity", "price"]
    assert "regime_new" in lines[0]
    assert len(lines) - 1 == room.observations
    assert room.observations > 0
    # XLSX — настоящий zip-контейнер, а не пустые байты.
    assert room.xlsx_bytes[:2] == b"PK"
    assert room.filename_stem == "round_01_ols_multiple"


async def test_data_room_is_deterministic_per_round(session: AsyncSession) -> None:
    """Два вызова — байт в байт одно и то же: разбор после раунда возможен."""
    from dashboard.actions import create_and_open_round, data_room

    round_ = await create_and_open_round(
        session,
        number=1,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="",
    )
    assert round_.id is not None
    first = await data_room(session, round_.id)
    second = await data_room(session, round_.id)
    assert first is not None and second is not None
    assert first.csv_text == second.csv_text
    assert first.brief_markdown == second.brief_markdown


async def test_data_room_none_for_missing_round(session: AsyncSession) -> None:
    from dashboard.actions import data_room

    assert await data_room(session, 999) is None


async def test_data_room_fails_loudly_without_teams_on_regime_shift(
    session: AsyncSession,
) -> None:
    """Раунд по режимному сдвигу без команд: данные не собираются, и это
    ValueError с объяснением, а не пустая комната. Страница показывает текст."""
    from dashboard.actions import create_and_open_round, data_room
    from db.enums import Method

    round_ = await create_and_open_round(
        session,
        number=1,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="",
        method=Method.OLS_MULTIPLE,
    )
    assert round_.id is not None
    with pytest.raises(ValueError, match="quantity_gap_ratio"):
        await data_room(session, round_.id)
