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


# --------------------------------------------------------------------------- #
# Метод раунда выбирается в форме, а не зашит в OLS_SIMPLE
# --------------------------------------------------------------------------- #


async def test_create_and_open_round_persists_chosen_method(
    session: AsyncSession,
) -> None:
    round_ = await create_and_open_round(
        session,
        number=2,
        difficulty=2,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="режимный сдвиг",
        method=Method.OLS_MULTIPLE,
    )
    assert round_.method is Method.OLS_MULTIPLE
    assert round_.status is RoundStatus.OPEN


async def test_create_and_open_round_rejects_method_without_case(
    session: AsyncSession,
) -> None:
    """Раунд по методу без кейса не создаётся — падать при создании, а не при
    выгрузке, когда команды уже ждут данные."""
    with pytest.raises(ValueError, match="autocorrelation"):
        await create_and_open_round(
            session,
            number=3,
            difficulty=1,
            market_a=100.0,
            market_b=1.0,
            market_mc=10.0,
            case_narrative="кейса нет",
            method=Method.AUTOCORRELATION,
        )
    # Черновик с неподдерживаемым методом не должен остаться в базе.
    assert await repo.list_rounds(session) == []


def test_method_choices_are_exactly_supported_cases() -> None:
    """Форма предлагает ровно те методы, под которые есть данные."""
    from core.cases import supported_methods
    from dashboard.actions import METHOD_CHOICES

    assert tuple(METHOD_CHOICES) == supported_methods()
    for method, label in METHOD_CHOICES.items():
        assert isinstance(method, Method)
        assert label and label != method.value


# --------------------------------------------------------------------------- #
# Панель разбора: во что верила команда и кто попался на ловушку
# --------------------------------------------------------------------------- #


async def _regime_round_with_three_teams(session: AsyncSession) -> tuple[int, list[int]]:
    """Раунд по режимному сдвигу (a=100, b=1, c=10) с тремя командами."""
    team_ids: list[int] = []
    for i in range(3):
        team = await repo.create_team(session, name=f"T{i}", company_name=f"C{i}")
        assert team.id is not None
        team_ids.append(team.id)
    round_ = await create_and_open_round(
        session,
        number=1,
        difficulty=2,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="",
        method=Method.OLS_MULTIPLE,
    )
    assert round_.id is not None
    return round_.id, team_ids


async def test_review_panel_hidden_until_round_closed(session: AsyncSession) -> None:
    """До закрытия разбор не показывается: он содержит истинный наклон."""
    from dashboard.actions import review_panel

    round_id, team_ids = await _regime_round_with_three_teams(session)
    await submit_manual_decision(
        session, team_id=team_ids[0], round_id=round_id, quantity=22.5, reasoning=""
    )
    assert await review_panel(session, round_id) is None


async def test_review_panel_flags_naive_team_and_clears_nash_team(
    session: AsyncSession,
) -> None:
    """Наивная команда — та, что сдала объём наивной процедуры на данных
    раунда; верные — объём верной. Вердикт по объёму, не по b̂ (12.09)."""
    from core.cases import REGIME_COLUMN
    from core.trap import Verdict, procedure_quantities
    from dashboard.actions import build_round_dataset, review_panel

    round_id, team_ids = await _regime_round_with_three_teams(session)
    dataset = await build_round_dataset(session, round_id)
    proc = procedure_quantities(
        dataset.rows,
        true_a=100.0,
        true_b=1.0,
        marginal_cost=10.0,
        n_firms=3,
        regime_column=REGIME_COLUMN,
    )
    quantities = {
        team_ids[0]: proc.sound_quantity,
        team_ids[1]: proc.sound_quantity,
        team_ids[2]: proc.naive_quantity,
    }
    for team_id, q in quantities.items():
        await submit_manual_decision(
            session, team_id=team_id, round_id=round_id, quantity=q, reasoning=""
        )
    await close_round_with_results(session, round_id)

    rows = await review_panel(session, round_id)
    assert rows is not None
    by_team = {row.team_name: row for row in rows}
    assert by_team["T2"].verdict is Verdict.TRAPPED
    assert by_team["T0"].verdict is Verdict.SOUND
    assert by_team["T1"].verdict is Verdict.SOUND
    assert by_team["T2"].naive_quantity == pytest.approx(proc.naive_quantity)
    assert by_team["T2"].sound_quantity == pytest.approx(proc.sound_quantity)
    # Наивная перепроизвела: ждала цену выше фактической, недобрала прибыль.
    assert by_team["T2"].price_gap > 0
    assert by_team["T2"].profit_gap > 0
    assert by_team["T2"].true_slope == 1.0
    assert by_team["T2"].naive_slope == pytest.approx(0.35)
    # Оценка раунда — доля от лучшего ответа: у наивной ниже, она сверху.
    assert by_team["T2"].br_share < by_team["T0"].br_share
    assert rows[0].team_name == "T2"


async def test_raw_profit_would_reward_the_trapped_team(session: AsyncSession) -> None:
    """Почему оценка — доля от лучшего ответа, а не сырая прибыль: в Курно
    единственный перепроизводитель зарабатывает больше тех, кто считал верно.
    Фиксируем факт числом, чтобы правило оценки не «упростили» обратно."""
    from core.cases import REGIME_COLUMN
    from core.trap import procedure_quantities
    from dashboard.actions import build_round_dataset, review_panel

    round_id, team_ids = await _regime_round_with_three_teams(session)
    dataset = await build_round_dataset(session, round_id)
    proc = procedure_quantities(
        dataset.rows,
        true_a=100.0,
        true_b=1.0,
        marginal_cost=10.0,
        n_firms=3,
        regime_column=REGIME_COLUMN,
    )
    for i, team_id in enumerate(team_ids):
        q = proc.naive_quantity if i == 2 else proc.sound_quantity
        await submit_manual_decision(
            session, team_id=team_id, round_id=round_id, quantity=q, reasoning=""
        )
    money = await close_round_with_results(session, round_id)
    assert money[0].team_name == "T2"  # по деньгам агрессор первый
    rows = await review_panel(session, round_id)
    assert rows is not None and rows[0].team_name == "T2"  # по оценке — последний


async def test_review_panel_no_trap_for_simple_regression(session: AsyncSession) -> None:
    from core.trap import Verdict
    from dashboard.actions import review_panel

    team = await repo.create_team(session, name="solo", company_name="S")
    assert team.id is not None
    round_id = await _make_round(session)
    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=45.0, reasoning=""
    )
    await close_round_with_results(session, round_id)
    rows = await review_panel(session, round_id)
    assert rows is not None and len(rows) == 1
    assert rows[0].verdict is Verdict.NO_TRAP
    assert rows[0].naive_slope is None
    # Монополист на Нэше: наклон истинный, разрывы нулевые.
    assert rows[0].implied_slope == pytest.approx(1.0)
    assert rows[0].profit_gap == pytest.approx(0.0, abs=1e-9)


async def test_ensure_rubric_seeds_default_for_regime_shift(session: AsyncSession) -> None:
    """Раунд по режимному сдвигу оценивается без ручного шаблона —
    рубрика по умолчанию заводится из core.rubrics и сохраняется."""
    from dashboard.actions import ensure_rubric_for_method

    rubric = await ensure_rubric_for_method(session, Method.OLS_MULTIPLE)
    assert any(c.id == "pooled_trap_named" for c in rubric)
    template = await repo.get_rubric_for_method(session, Method.OLS_MULTIPLE)
    assert template is not None


async def test_ensure_rubric_still_refuses_method_without_default(
    session: AsyncSession,
) -> None:
    from dashboard.actions import ensure_rubric_for_method

    with pytest.raises(ValueError, match="autocorrelation"):
        await ensure_rubric_for_method(session, Method.AUTOCORRELATION)


async def test_defence_draw_only_after_close_and_only_among_deciders(
    session: AsyncSession,
) -> None:
    from dashboard.actions import defence_draw

    round_id, team_ids = await _regime_round_with_three_teams(session)
    # Решение подала только одна команда — её и разыгрываем.
    await submit_manual_decision(
        session, team_id=team_ids[1], round_id=round_id, quantity=22.5, reasoning=""
    )
    assert await defence_draw(session, round_id) is None  # раунд ещё открыт
    await close_round_with_results(session, round_id)

    card = await defence_draw(session, round_id)
    assert card is not None
    assert card.team_name == "T1"
    assert card.question.text
    # Повторный вызов — та же карточка; перетяжка — воспроизводима.
    assert await defence_draw(session, round_id) == card
    redraw = await defence_draw(session, round_id, attempt=1)
    assert redraw is not None and redraw.attempt == 1
    assert redraw == await defence_draw(session, round_id, attempt=1)


# --------------------------------------------------------------------------- #
# market_preview: подсказки преподавателю перед созданием раунда
# --------------------------------------------------------------------------- #


async def test_market_preview_computes_nash_for_team_count(session: AsyncSession) -> None:
    from dashboard.actions import market_preview

    for i in range(3):
        await repo.create_team(session, name=f"T{i}", company_name=f"C{i}")
    preview = await market_preview(session, market_a=100.0, market_b=1.0, market_mc=10.0)
    assert preview.n_firms == 3
    assert preview.nash_quantity == pytest.approx(22.5)
    assert preview.nash_price == pytest.approx(32.5)
    assert preview.warnings == ()


async def test_market_preview_warns_on_repeated_parameters(session: AsyncSession) -> None:
    from dashboard.actions import market_preview

    await _make_round(session)  # a=100, b=1, c=10
    preview = await market_preview(session, market_a=100.0, market_b=1.0, market_mc=10.0)
    assert any("совпадают" in w for w in preview.warnings)
    changed = await market_preview(session, market_a=120.0, market_b=1.0, market_mc=10.0)
    assert not any("совпадают" in w for w in changed.warnings)


async def test_market_preview_warns_on_tight_market(session: AsyncSession) -> None:
    """Семь команд на 100/1/15: цена Нэша 25.6, полоса 10.6 > 0.25·25.6 — нет;
    берём 100/1/20: цена 30, полоса 10 < 7.5? нет. Считаем честно ниже."""
    from dashboard.actions import market_preview

    for i in range(7):
        await repo.create_team(session, name=f"T{i}", company_name=f"C{i}")
    # a=100, b=1, c=40 при семи: q = 60/8 = 7.5, P = 100 - 52.5 = 47.5,
    # полоса 7.5 < 0.25 · 47.5 = 11.9 → тесный.
    tight = await market_preview(session, market_a=100.0, market_b=1.0, market_mc=40.0)
    assert any("тесный" in w for w in tight.warnings)
    wide = await market_preview(session, market_a=300.0, market_b=1.0, market_mc=10.0)
    assert not any("тесный" in w for w in wide.warnings)


async def test_second_open_round_is_refused(session: AsyncSession) -> None:
    """Один открытый раунд за раз: иначе бот шлёт решения не туда.
    Черновик второго раунда в базе не остаётся."""
    first = await _make_round(session)
    with pytest.raises(ValueError, match="ещё открыт"):
        await create_and_open_round(
            session,
            number=2,
            difficulty=1,
            market_a=120.0,
            market_b=1.0,
            market_mc=10.0,
            case_narrative="",
        )
    rounds = await repo.list_rounds(session)
    assert [r.id for r in rounds] == [first]
    # После закрытия первого второй открывается.
    await repo.set_round_status(session, round_id=first, status=RoundStatus.CLOSED)
    second = await create_and_open_round(
        session,
        number=2,
        difficulty=1,
        market_a=120.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="",
    )
    assert second.status is RoundStatus.OPEN
