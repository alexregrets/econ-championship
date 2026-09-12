"""Заполнить базу для показа: команды, сыгранный раунд, разбор — одной командой.

    uv run python -m devshell.demo_round            # seed + раунд 2 закрыт
    uv run python -m devshell.demo_round --grade    # плюс Groq-оценка обоснований

Зачем. Демонстрация с пустого экрана — это «а теперь представьте». Скрипт
делает то, что за неделю сделали бы студенты: семь команд вступают, получают
брифинг, шесть считают правильно, одна скидывает данные в чат и получает
наивный ответ — и подаёт его. Раунд закрыт, разбор готов: на странице
раундов видно, кто попался, `/result` у команд заполнен, микрозащиту можно
разыграть кнопкой.

Скрипт **пересоздаёт базу** (`devshell.seed`). На боевой базе не запускать.
Рынок берётся шире сидового (`a=300`), чтобы одна наивная команда не роняла
цену в ноль — тесный рынок делает разбор бойней, а не уроком.
"""

from __future__ import annotations

import argparse
import asyncio
from functools import partial

from bot.service import join_team, submit_decision
from dashboard.actions import (
    build_grading_llm,
    close_round_with_results,
    create_and_open_round,
    data_room,
    grade_round_reasoning,
    market_preview,
    review_panel,
)
from dashboard.db_runner import run_db
from db import bot_repositories as bot_repo
from db import repositories as repo
from db.enums import Method, RoundStatus
from devshell.seed import reset_and_seed

# Обоснование команды, которая считала сама: числа из данных, ловушка названа.
_SOUND_REASONING = (
    "В выборке два режима (regime_new). Сквозная регрессия P на Q даёт наклон около "
    "{naive:.2f} — заниженный: кластеры двух режимов тянут линию в пологую, по ней мы "
    "бы перепроизвели. С дамми и взаимодействием regime_new*Q наклон текущего режима "
    "{b:.2f} (старый примерно {b_old:.2f}), R2 около 0.9. Издержки c={c:g}. Ожидаем "
    "суммарный выпуск соперников около {rivals:.1f}, остаточный спрос A = {a:g} - "
    "{b:g}*{rivals:.1f} = {residual:.1f}, по условию первого порядка q = ({residual:.1f} - "
    "{c:g}) / (2*{b:g}) = {q:.2f}; ожидаемая цена около {price:.1f}."
)
# Обоснование команды, которая скинула CSV в чат и вставила ответ.
_GENERIC_REASONING = (
    "Мы оценили спрос методом наименьших квадратов. Наклон отрицательный и "
    "статистически значимый. На основе полученной модели мы выбрали оптимальный "
    "объём выпуска, максимизирующий прибыль компании с учётом конкуренции."
)

_A, _B, _C = 300.0, 1.0, 30.0


def _pooled_ols(rows: list[dict[str, float]]) -> tuple[float, float]:
    """Сквозная регрессия цены на выпуск по всей истории: (â, b̂), b̂ > 0.

    Ровно то, что делает команда, которая не заметила два режима: одна
    линия через оба кластера. Чистый Python — devshell не тянет numpy.
    """
    n = len(rows)
    mean_q = sum(r["quantity"] for r in rows) / n
    mean_p = sum(r["price"] for r in rows) / n
    cov = sum((r["quantity"] - mean_q) * (r["price"] - mean_p) for r in rows)
    var = sum((r["quantity"] - mean_q) ** 2 for r in rows)
    slope = cov / var  # отрицательный: цена падает с выпуском
    return mean_p - slope * mean_q, -slope


def main(grade: bool) -> None:
    """Собрать демо-состояние. ``grade`` — дополнительно оценить обоснования Groq."""
    asyncio.run(reset_and_seed())
    teams = run_db(repo.list_teams)
    codes = run_db(bot_repo.ensure_join_codes)

    # Сидовый раунд 1 закрываем без решений — он нужен только как история.
    first = run_db(repo.get_open_round)
    assert first is not None and first.id is not None
    run_db(partial(repo.set_round_status, round_id=first.id, status=RoundStatus.CLOSED))

    preview = run_db(partial(market_preview, market_a=_A, market_b=_B, market_mc=_C))
    round_ = run_db(
        partial(
            create_and_open_round,
            number=2,
            difficulty=2,
            market_a=_A,
            market_b=_B,
            market_mc=_C,
            case_narrative="",
            method=Method.OLS_MULTIPLE,
        )
    )
    assert round_.id is not None
    n = len(teams)
    nash = preview.nash_quantity
    rivals = nash * (n - 1)
    residual = _A - _B * rivals
    # Наивная команда: сквозная линия по данным раунда и лучший ответ по ней.
    room = run_db(partial(data_room, round_id=round_.id))
    assert room is not None
    header, *body = room.csv_text.strip().splitlines()
    names = header.split(",")
    rows_csv = [dict(zip(names, map(float, line.split(",")), strict=True)) for line in body]
    a_hat, b_hat = _pooled_ols(rows_csv)
    naive_q = max((a_hat - b_hat * rivals - _C) / (2 * b_hat), 0.0)
    print(f"Наивная сквозная линия: P = {a_hat:.1f} - {b_hat:.3f}·Q (истина: {_A:g} - {_B:g}·Q)")
    sound = _SOUND_REASONING.format(
        naive=-0.35 * _B,
        b=-_B,
        b_old=-0.5 * _B,
        c=_C,
        rivals=rivals,
        a=_A,
        residual=residual,
        q=nash,
        price=preview.nash_price,
    )

    for i, team in enumerate(teams):
        assert team.id is not None
        run_db(
            partial(
                join_team,
                telegram_id=100_000 + i,
                full_name=f"Студент {i + 1}",
                code=codes[team.id],
            )
        )
        naive = i == n - 1  # последняя команда — та, что скинула всё в чат
        run_db(
            partial(
                submit_decision,
                telegram_id=100_000 + i,
                quantity=naive_q if naive else nash,
                reasoning=_GENERIC_REASONING if naive else sound,
            )
        )

    rows = run_db(partial(close_round_with_results, round_id=round_.id))
    print(f"Раунд №{round_.number} ({round_.method.value}) закрыт, {n} команд:")
    for row in rows:
        print(f"  {row.team_name:12} Q={row.quantity:7.2f} прибыль={row.market_score:9.1f}")
    review = run_db(partial(review_panel, round_id=round_.id)) or []
    flagged = [r.team_name for r in review if r.verdict.value == "trapped"]
    print("Попались на ловушку:", ", ".join(flagged) or "никто")

    if grade:
        llm = build_grading_llm()
        graded = run_db(partial(grade_round_reasoning, round_id=round_.id, llm=llm))
        for row in graded:
            print(f"  рубрика {row.team_name:12} {row.rubric_score:.2f}")
    print("Готово: откройте дашборд, страница «Раунды».")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grade", action="store_true", help="оценить обоснования Groq")
    main(grade=parser.parse_args().grade)
