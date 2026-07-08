# Развёртывание к сентябрьскому запуску

Как поднять бота и дашборд на сервере. Написано под самый дешёвый VPS
(1 vCPU / 512 МБ, Ubuntu 22.04/24.04 LTS) — для 21 команды этого достаточно
с большим запасом. Всё то же самое работает и на ноутбуке преподавателя:
тогда раздел про systemd можно пропустить и просто держать два терминала.

## Что где живёт

| Процесс | Как ходит наружу | Кому нужен доступ |
|---|---|---|
| Telegram-бот (`python -m bot.main`) | **long polling**: бот сам ходит к Telegram, публичный IP/домен/HTTPS **не нужны** | студенты (через Telegram) |
| Дашборд (`streamlit run dashboard/app.py`) | слушает локальный порт | только преподаватель |
| База | один файл SQLite `econ_tournament.db` | оба процесса читают его же |

Из-за общего файла SQLite бот и дашборд должны работать **на одной машине**
и из **одной директории** (путь к базе — `DATABASE_URL` в `.env`,
по умолчанию `./econ_tournament.db` относительно рабочей директории).

## Установка (один раз)

```bash
# 1. Код и зависимости
sudo mkdir -p /opt/econ-tournament && sudo chown $USER /opt/econ-tournament
git clone <репозиторий> /opt/econ-tournament
cd /opt/econ-tournament
curl -LsSf https://astral.sh/uv/install.sh | sh   # установит uv в ~/.local/bin
uv sync                                           # скачает Python 3.12 и зависимости
uv run pytest -q                                  # все тесты зелёные — окружение исправно

# 2. Секреты
cp .env.example .env
nano .env    # BOT_TOKEN=... (от @BotFather), GROQ_API_KEY=... (для грейдинга)

# 3. Данные турнира
uv run python -m devshell.seed          # команды + первый открытый раунд
uv run python -m devshell.team_codes    # печатает /join-коды — раздать командам
```

Важно: «Seed» **пересоздаёт базу с нуля**. На боевом сервере запускать его
только при первичной настройке, не между раундами.

## Бот как systemd-сервис (авторестарт после падений и ребута)

`/etc/systemd/system/econ-bot.service`:

```ini
[Unit]
Description=Econometrics Championship Telegram bot
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/econ-tournament
ExecStart=/home/ubuntu/.local/bin/uv run python -m bot.main
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

(`User`, `WorkingDirectory` и путь к `uv` поправить под свой сервер.)

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now econ-bot
journalctl -u econ-bot -f        # живые логи
```

**Ровно один экземпляр бота.** Telegram отдаёт обновления через `getUpdates`
только одному процессу: второй запущенный бот (например, тестовый на
ноутбуке с тем же токеном) вызовет конфликт и «зависшие» сообщения.
Останавливайте локального бота перед запуском серверного.

## Дашборд — только через SSH-туннель

У Streamlit нет встроенной авторизации, а дашборд умеет закрывать раунды —
наружу его не открываем. Запуск на сервере:

```bash
cd /opt/econ-tournament
uv run streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

На машине преподавателя:

```bash
ssh -L 8501:127.0.0.1:8501 ubuntu@<сервер>
# дальше в браузере: http://localhost:8501
```

Дашборд нужен только в моменты закрытия раунда и оценки обоснований, поэтому
держать его сервисом не обязательно — можно запускать по необходимости в
tmux/screen. Если всё же хочется постоянно — второй systemd-юнит по образцу
бота с `ExecStart=... uv run streamlit run dashboard/app.py --server.address 127.0.0.1`.

## Бэкапы: один файл — одна команда

Вся история турнира лежит в `econ_tournament.db`. Перед закрытием каждого
раунда (и после — по вкусу):

```bash
cp econ_tournament.db "backup/econ_$(date +%F_%H-%M).db"
```

Автоматически — раз в час в cron: `0 * * * * cd /opt/econ-tournament && cp econ_tournament.db backup/econ_$(date +\%F_\%H).db`.
Восстановление — остановить бота, подложить копию на место, запустить.

## Обновление кода

```bash
cd /opt/econ-tournament
git pull
uv sync                       # подтянуть новые зависимости из uv.lock
uv run pytest -q              # убедиться, что всё зелёное
sudo systemctl restart econ-bot
```

`init_db()` при старте бота создаёт **недостающие** таблицы сам; переименования
существующих колонок потребуют миграции руками (пока таких изменений не было).

## Чеклист перед первой парой

1. `systemctl status econ-bot` — active (running).
2. С личного Telegram: `/start`, `/join <код>`, `/submit 1 тест`, `/status` —
   бот отвечает; тестовое решение затем перезаписать или закрыть раунд заново
   не забыть.
3. Дашборд через туннель открывается, раунд виден, счётчик решений растёт.
4. `GROQ_API_KEY` в `.env` — кнопка «Оценить обоснования» работает
   (проверяется только на закрытом раунде).
5. Свежий бэкап базы сделан.
