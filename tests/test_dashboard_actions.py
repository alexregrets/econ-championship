"""Тесты действий дашборда (dashboard/actions.py).

Тем же стилем, что tests/test_round_service.py: in-memory база, никакого
Streamlit и никакой сети — проверяем ровно ту логику, которую страница
вызывает по кнопкам, с конкретными числовыми проверками.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.rubric_grader import GradingResult, RubricCriterion
from dashboard.actions import (
    close_round_with_results,
    create_and_open_round,
    grade_round_reasoning,
    next_round_number,
    results_table,
    submit_manual_decision,
)
from db import models  # noqa: F401  (регистрирует таблицы)
from db import repositories as repo
from db.enums import EngineMode, Method, RoundStatus
from llm.base import LLMError


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _make_round(session: AsyncSession) -> int:
    """Создать и открыть стандартный раунд-«нефть» для тестов, вернуть id."""
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


async def test_next_round_number_starts_at_one(session: AsyncSession) -> None:
    assert await next_round_number(session) == 1


async def test_next_round_number_increments(session: AsyncSession) -> None:
    await _make_round(session)
    assert await next_round_number(session) == 2


async def test_create_and_open_round_is_open_and_ols_simple(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    round_ = await repo.get_round(session, round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN
    assert round_.method is Method.OLS_SIMPLE
    assert round_.market_a == 100.0
    # Без явного engine_mode раунд симметричный — дефолт формы не сдвинут.
    assert round_.engine_mode is EngineMode.SYMMETRIC


async def test_create_and_open_round_asymmetric_mode_persists(
    session: AsyncSession,
) -> None:
    round_ = await create_and_open_round(
        session,
        number=1,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="асимметричный пилот",
        engine_mode=EngineMode.ASYMMETRIC,
    )
    assert round_.engine_mode is EngineMode.ASYMMETRIC


async def test_submit_manual_decision_creates_and_revises(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="Команда А", company_name="Роснефть")
    assert team.id is not None

    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=20.0, reasoning="v1"
    )
    # Повторная отправка перезаписывает, дубликата не появляется.
    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=30.0, reasoning="v2"
    )
    decisions = await repo.list_decisions_for_round(session, round_id)
    assert len(decisions) == 1
    assert decisions[0].quantity == 30.0
    assert decisions[0].reasoning == "v2"


async def test_submit_manual_decision_rejects_closed_round(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="Команда А", company_name="Роснефть")
    assert team.id is not None
    await repo.set_round_status(
        session, round_id=round_id, status=RoundStatus.CLOSED
    )
    with pytest.raises(ValueError):
        await submit_manual_decision(
            session, team_id=team.id, round_id=round_id, quantity=5.0, reasoning=""
        )


async def test_close_round_with_results_full_flow(session: AsyncSession) -> None:
    """Дуополия с известным ответом: q1=30, q2=20 → P=50, прибыли 1200 и 800."""
    round_id = await _make_round(session)
    team_a = await repo.create_team(session, name="А", company_name="Роснефть")
    team_b = await repo.create_team(session, name="Б", company_name="Газпром")
    assert team_a.id is not None and team_b.id is not None

    await submit_manual_decision(
        session, team_id=team_a.id, round_id=round_id, quantity=30.0, reasoning="x"
    )
    await submit_manual_decision(
        session, team_id=team_b.id, round_id=round_id, quantity=20.0, reasoning="y"
    )

    rows = await close_round_with_results(session, round_id)

    # Раунд закрыт, решения больше не принимаются.
    round_ = await repo.get_round(session, round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.CLOSED

    # P = 100 - 1*(30+20) = 50; прибыль = (P - mc) * q.
    assert len(rows) == 2
    assert rows[0].team_name == "А"  # сортировка по прибыли: 1200 сверху
    assert rows[0].price == pytest.approx(50.0)
    assert rows[0].market_score == pytest.approx((50.0 - 10.0) * 30.0)
    assert rows[1].market_score == pytest.approx((50.0 - 10.0) * 20.0)
    # LLM-грейдинг к сервису не подключён — rubric_score дефолтный.
    assert rows[0].rubric_score == 0.0


async def test_results_table_skips_unscored_decisions(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="А", company_name="Роснефть")
    assert team.id is not None
    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=10.0, reasoning=""
    )
    # Раунд не закрывали — Result ещё не посчитан, таблица должна быть пустой.
    assert await results_table(session, round_id) == []


# --------------------------------------------------------------------------- #
# grade_round_reasoning: LLM-грейдинг закрытого раунда (mock LLM, без сети)
# --------------------------------------------------------------------------- #


class _FakeGradingLLM:
    """Фейковый StructuredLLM: «зачитывает» ровно заданные критерии.

    Возвращает passed=True для критериев из ``passed_ids`` и ничего для
    остальных — по семантике грейдера пропущенные считаются проваленными.
    """

    def __init__(self, passed_ids: set[str]) -> None:
        self.passed_ids = passed_ids
        self.calls = 0

    async def structured_completion[T: BaseModel](
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        self.calls += 1
        grades = [
            {"criterion_id": cid, "passed": True, "evidence": "цитата"}
            for cid in sorted(self.passed_ids)
        ]
        return response_model(grades=grades)


class _ExplodingLLM:
    """Фейковый StructuredLLM, имитирующий недоступный Groq."""

    async def structured_completion[T: BaseModel](
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        raise LLMError("groq недоступен (fake)")


async def _closed_duopoly_round(session: AsyncSession) -> int:
    """Подготовить закрытый раунд-дуополию с обоснованиями, вернуть id."""
    round_id = await _make_round(session)
    team_a = await repo.create_team(session, name="А", company_name="Роснефть")
    team_b = await repo.create_team(session, name="Б", company_name="Газпром")
    assert team_a.id is not None and team_b.id is not None
    await submit_manual_decision(
        session,
        team_id=team_a.id,
        round_id=round_id,
        quantity=30.0,
        reasoning="Оценили спрос парной регрессией, наклон отрицательный.",
    )
    await submit_manual_decision(
        session,
        team_id=team_b.id,
        round_id=round_id,
        quantity=20.0,
        reasoning="Q выбрали из ожидаемой цены по модели.",
    )
    await close_round_with_results(session, round_id)
    return round_id


async def test_grade_round_writes_real_rubric_scores(
    session: AsyncSession,
) -> None:
    """Проходят specification (0.3) и interpretation (0.3) → score 0.6."""
    round_id = await _closed_duopoly_round(session)
    llm = _FakeGradingLLM({"specification", "interpretation"})

    rows = await grade_round_reasoning(session, round_id, llm)

    assert llm.calls == 2  # по одному вызову на решение
    assert len(rows) == 2
    for row in rows:
        assert row.rubric_score == pytest.approx(0.6)
    # Рыночные числа не тронуты: победитель по-прежнему с прибылью 1200.
    assert rows[0].market_score == pytest.approx(1200.0)
    assert rows[0].price == pytest.approx(50.0)

    # Полный GradingResult сохранён для аудита и содержит все 4 критерия.
    decisions = await repo.list_decisions_for_round(session, round_id)
    assert decisions[0].id is not None
    stored = await repo.get_result_for_decision(session, decisions[0].id)
    assert stored is not None
    breakdown = GradingResult.model_validate_json(stored.rubric_breakdown_json)
    assert len(breakdown.grades) == 4
    assert breakdown.total_score == pytest.approx(0.6)

    # Пилотная рубрика заведена в БД как RubricTemplate.
    template = await repo.get_rubric_for_method(session, Method.OLS_SIMPLE)
    assert template is not None
    assert "specification" in template.criteria_json


async def test_grade_round_requires_closed_round(session: AsyncSession) -> None:
    round_id = await _make_round(session)  # раунд открыт
    with pytest.raises(ValueError):
        await grade_round_reasoning(session, round_id, _FakeGradingLLM(set()))


async def test_grade_failure_leaves_market_results_intact(
    session: AsyncSession,
) -> None:
    """Groq упал → LLMError наружу, рыночные результаты и score не тронуты."""
    round_id = await _closed_duopoly_round(session)
    with pytest.raises(LLMError):
        await grade_round_reasoning(session, round_id, _ExplodingLLM())

    rows = await results_table(session, round_id)
    assert len(rows) == 2
    assert rows[0].market_score == pytest.approx(1200.0)
    assert all(row.rubric_score == 0.0 for row in rows)


async def test_grade_respects_existing_rubric_template(
    session: AsyncSession,
) -> None:
    """Свой RubricTemplate в БД имеет приоритет над пилотной рубрикой."""
    round_id = await _closed_duopoly_round(session)
    custom = [RubricCriterion(id="only", description="один критерий", weight=1.0)]
    await repo.upsert_rubric_template(
        session,
        method=Method.OLS_SIMPLE,
        name="Своя рубрика",
        criteria_json=f'[{custom[0].model_dump_json()}]',
    )

    rows = await grade_round_reasoning(session, round_id, _FakeGradingLLM({"only"}))
    assert all(row.rubric_score == pytest.approx(1.0) for row in rows)

    # Свой шаблон не перезаписан пилотным.
    template = await repo.get_rubric_for_method(session, Method.OLS_SIMPLE)
    assert template is not None
    assert template.name == "Своя рубрика"


async def test_grade_rerun_overwrites_scores(session: AsyncSession) -> None:
    """Повторная оценка перезаписывает score — операция идемпотентна."""
    round_id = await _closed_duopoly_round(session)
    all_ids = {"specification", "interpretation", "quantity_link", "fit_check"}

    rows = await grade_round_reasoning(session, round_id, _FakeGradingLLM(all_ids))
    assert all(row.rubric_score == pytest.approx(1.0) for row in rows)

    rows = await grade_round_reasoning(session, round_id, _FakeGradingLLM(set()))
    assert all(row.rubric_score == 0.0 for row in rows)
