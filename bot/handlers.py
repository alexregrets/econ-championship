"""aiogram-хендлеры: парсинг команд и форматирование ответов, ничего больше.

Вся работа с БД — в :mod:`bot.service`; здесь только разбор аргументов,
человеческие тексты и перевод ValueError в ответное сообщение. Каждое
обновление обрабатывается в собственной сессии через ``get_session_ctx``
(бот живёт в одном event loop, глобальный движок переиспользуется безопасно —
в отличие от Streamlit, см. DECISIONS.md №3).
"""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

from bot.service import join_team, submit_decision, team_status
from db.session import get_session_ctx

__all__ = ["router"]

router = Router(name="tournament")

_START_TEXT = (
    "Это бот чемпионата по эконометрике.\n\n"
    "Команды:\n"
    "/join <код> — вступить в свою команду (код выдаёт преподаватель)\n"
    "/submit <Q> <обоснование> — подать решение команды в открытый раунд\n"
    "/status — команда, открытый раунд и поданное решение\n\n"
    "Пока раунд открыт, /submit можно повторять — учитывается последнее."
)

_JOIN_USAGE = "Использование: /join <код команды>"
_SUBMIT_USAGE = (
    "Использование: /submit <Q> <обоснование>\n"
    "Например: /submit 120 Оценили спрос по цене Urals, наклон "
    "отрицательный, поэтому не наращиваем добычу."
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Поздороваться и показать список команд."""
    await message.answer(_START_TEXT)


@router.message(Command("join"))
async def cmd_join(message: Message, command: CommandObject) -> None:
    """Вступить в команду по коду: /join <код>."""
    user = message.from_user
    if user is None:  # каналы/анонимные посты — не наш сценарий
        return
    code = (command.args or "").strip()
    if not code:
        await message.answer(_JOIN_USAGE)
        return

    async with get_session_ctx() as session:
        try:
            outcome = await join_team(
                session,
                telegram_id=user.id,
                full_name=user.full_name,
                code=code,
            )
        except ValueError as exc:
            await message.answer(str(exc))
            return

    greeting = "Готово" if outcome.created_student else "Перепривязал"
    await message.answer(
        f"{greeting}: вы в команде «{outcome.team.name}» "
        f"({outcome.team.company_name}).\n"
        "Когда раунд открыт, подавайте решение: /submit <Q> <обоснование>."
    )


@router.message(Command("submit"))
async def cmd_submit(message: Message, command: CommandObject) -> None:
    """Подать решение: /submit <Q> <обоснование>."""
    user = message.from_user
    if user is None:
        return
    args = (command.args or "").strip()
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(_SUBMIT_USAGE)
        return
    quantity_raw, reasoning = parts
    try:
        # Студенты пишут и «120,5», и «120.5» — принимаем оба варианта.
        quantity = float(quantity_raw.replace(",", "."))
    except ValueError:
        await message.answer(
            f"Не понял объём «{quantity_raw}» — нужно число.\n{_SUBMIT_USAGE}"
        )
        return

    async with get_session_ctx() as session:
        try:
            outcome = await submit_decision(
                session,
                telegram_id=user.id,
                quantity=quantity,
                reasoning=reasoning,
            )
        except ValueError as exc:
            await message.answer(str(exc))
            return

    verb = "заменил прежнее" if outcome.replaced else "принял"
    await message.answer(
        f"Раунд №{outcome.round.number}: {verb} решение команды "
        f"«{outcome.team.name}» — Q = {outcome.decision.quantity:g}.\n"
        "Пока раунд открыт, /submit можно повторить — учитывается последнее."
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    """Показать команду, открытый раунд и поданное решение."""
    user = message.from_user
    if user is None:
        return

    async with get_session_ctx() as session:
        try:
            status = await team_status(session, telegram_id=user.id)
        except ValueError as exc:
            await message.answer(str(exc))
            return

    lines = [f"Команда: «{status.team.name}» ({status.team.company_name})."]
    if status.open_round is None:
        lines.append("Открытого раунда сейчас нет.")
    else:
        lines.append(f"Открыт раунд №{status.open_round.number}.")
        if status.decision is None:
            lines.append("Решение ещё не подано: /submit <Q> <обоснование>.")
        else:
            lines.append(
                f"Подано решение: Q = {status.decision.quantity:g} "
                f"(можно заменить повторным /submit)."
            )
    await message.answer("\n".join(lines))
