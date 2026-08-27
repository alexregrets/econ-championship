"""Раунд 4 — гетероскедастичность. Спецификация, написанная до кода.

Этот кейс устроен иначе, чем раунд 2, и в этом вся его ценность
(``CASES.md``): гетероскедастичность **не смещает** точечную оценку. Она
занижает стандартные ошибки. Наивная команда смотрит на t-статистику, видит
«значимо», принимает шаткую оценку за точную и ставит агрессивно. Правильный
вывод здесь — не другое число, а **другое поведение**.

Отсюда состав проверок. Ни одна из них не про «функция вернула список»:

1. Гетероскедастичность действительно есть — её отвергают Уайт и Бройш–Паган.
2. На baseline-раунде те же тесты молчат. Без этого контроля первая проверка
   ничего не стоит: тест, который срабатывает всегда, не срабатывает никогда.
3. Точечная оценка остаётся несмещённой — иначе это был бы другой кейс.
4. Классический доверительный интервал **недокрывает**: истинный наклон
   вылетает из номинального 95 %-го интервала заметно чаще, чем в 5 %
   случаев. Это и есть «стандартные ошибки занижены», переведённое в число.
5. Деньгами раунд **не** наказывает — проверено и зафиксировано тестом.
   ``CASES.md`` обещал обратное; замер 27 августа обещание опроверг, и кейс
   оценивается рубрикой метода, а не прибылью. Подробности — в самом тесте.

Свойства статистические, поэтому проверяются на многих зёрнах. Одно зерно
здесь не доказывает ничего — ни в одну, ни в другую сторону.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

from core.cases import DEFAULT_HETEROSKEDASTICITY, CaseData, build_case
from core.dataset import HistorySpec
from core.market_engine import MarketParameters
from db.enums import Method
from services.dataset_export import DEFAULT_PERIODS, _scaled_noise_and_spread

PARAMS = MarketParameters(a=2000.0, b=3.6, marginal_cost=364.0)
N_FIRMS = 4

# Сколько зёрен берётся под статистические свойства. Тридцать хватает, чтобы
# доля покрытия отличалась от номинала заметно, и не превращает прогон
# тестов в вычислительный эксперимент.
SEEDS = range(1, 31)


def _spec(seed: int, n_firms: int = N_FIRMS) -> HistorySpec:
    """Настройки ровно те же, что у боевой выгрузки."""
    return HistorySpec(
        periods=DEFAULT_PERIODS,
        seed=seed,
        **_scaled_noise_and_spread(PARAMS, n_firms),
    )


def _reference_quantity(n_firms: int = N_FIRMS) -> float:
    return n_firms * (PARAMS.a - PARAMS.marginal_cost) / ((n_firms + 1) * PARAMS.b)


def _case(seed: int, method: Method = Method.HETEROSCEDASTICITY) -> CaseData:
    return build_case(
        method,
        PARAMS,
        reference_quantity=_reference_quantity(),
        spec=_spec(seed),
    )


def _design(case: CaseData) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть матрицу регрессоров с константой и вектор цен."""
    quantities = np.array([o.quantity for o in case.observations])
    prices = np.array([o.price for o in case.observations])
    return sm.add_constant(quantities), prices


def _white_pvalue(case: CaseData) -> float:
    exog, prices = _design(case)
    fit = sm.OLS(prices, exog).fit()
    return float(het_white(fit.resid, exog)[1])


# --------------------------------------------------------------------------- #
# Форма кейса
# --------------------------------------------------------------------------- #


def test_case_is_deterministic() -> None:
    """Одно зерно — один датасет, иначе разбор после раунда невозможен."""
    assert _case(11).observations == _case(11).observations


def test_history_is_longer_than_usual() -> None:
    """История этого раунда длиннее обычной, и это требование метода.

    На тридцати наблюдениях Уайт ловил подложенный эффект в 60–80 % случаев:
    команда честно применяла метод и на своих данных ничего не находила.
    Разговор о дисперсии стоит дороже разговора о среднем, и оплачен он
    длиной истории, а не силой эффекта.
    """
    expected = DEFAULT_PERIODS * DEFAULT_HETEROSKEDASTICITY.periods_multiplier
    assert len(_case(11).observations) == expected
    assert [o.period for o in _case(11).observations] == list(range(1, expected + 1))


def test_case_adds_no_columns() -> None:
    """Гетероскедастичность живёт в остатках, а не в отдельном столбце.

    Столбец «дисперсия» выдал бы ответ до вопроса: команда должна увидеть
    неоднородность разброса сама, тестом.
    """
    assert _case(11).extra_columns == ()


# --------------------------------------------------------------------------- #
# Закономерность действительно заложена
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_white_rejects_homoskedasticity(seed: int) -> None:
    """Тест Уайта обязан отвергать гомоскедастичность на каждом зерне.

    Не «в среднем по зёрнам»: раунд играется один раз, и команда, честно
    применившая метод, обязана увидеть проблему на своих данных, а не на
    статистике по чужим.
    """
    assert _white_pvalue(_case(seed)) < 0.05


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_breusch_pagan_agrees(seed: int) -> None:
    """Бройш–Паган приходит к тому же выводу, что и Уайт.

    Оба теста в программе раунда, и расхождение между ними на учебных данных
    сбивало бы студента без всякой пользы.
    """
    exog, prices = _design(_case(seed))
    fit = sm.OLS(prices, exog).fit()
    assert float(het_breuschpagan(fit.resid, exog)[1]) < 0.05


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_baseline_round_is_not_flagged(seed: int) -> None:
    """На baseline-раунде Уайт молчит — контроль на ложное срабатывание.

    Без этой проверки предыдущая ничего не доказывает: тест, который
    отвергает всегда, не отличает кейс от чего угодно.
    """
    baseline = _case(seed, method=Method.OLS_SIMPLE)
    assert _white_pvalue(baseline) > 0.05


# --------------------------------------------------------------------------- #
# Что метод меняет, а что нет
# --------------------------------------------------------------------------- #


def test_point_estimate_stays_unbiased() -> None:
    """Точечная оценка наклона не смещается — это не ловушка раунда 2.

    Гетероскедастичность бьёт по эффективности и по стандартным ошибкам,
    а не по самой оценке. Если бы наклон уезжал систематически, кейс учил бы
    не тому, что заявлено.
    """
    slopes = []
    for seed in SEEDS:
        exog, prices = _design(_case(seed))
        slopes.append(-float(sm.OLS(prices, exog).fit().params[1]))

    assert float(np.mean(slopes)) == pytest.approx(PARAMS.b, rel=0.05)


def test_classical_interval_undercovers_and_robust_does_better() -> None:
    """Номинальный 95 %-й интервал держит заметно меньше 95 %.

    Это и есть «стандартные ошибки занижены», сказанное числом: команда
    строит интервал по классическим ошибкам, видит узкую вилку и верит ей
    больше, чем следует. Робастные ошибки (HC3) обязаны покрывать лучше.
    """
    classical_hits = 0
    robust_hits = 0

    for seed in SEEDS:
        exog, prices = _design(_case(seed))
        fit = sm.OLS(prices, exog).fit()
        robust = fit.get_robustcov_results(cov_type="HC3")

        for model, counter in ((fit, "classical"), (robust, "robust")):
            low, high = model.conf_int()[1]
            # Спрос убывает: наклон регрессии равен −b.
            if low <= -PARAMS.b <= high:
                if counter == "classical":
                    classical_hits += 1
                else:
                    robust_hits += 1

    total = len(SEEDS)
    assert classical_hits / total < 0.85
    assert robust_hits > classical_hits


def test_this_round_is_not_decided_by_money() -> None:
    """Деньгами этот раунд не наказывает — и это зафиксировано намеренно.

    ``CASES.md`` в первой редакции обещал обратное: «здесь оно измеряется
    деньгами». Замер 27 августа показал, что нет. Агрессивная команда,
    поставившая по точечной оценке, забирает практически всю оптимальную
    прибыль даже в худшем из сорока зёрен, а осторожная — на несколько
    процентов меньше: страховка дороже риска, от которого страхует.

    Причина не в калибровке. Прибыль вблизи оптимума плоская — теряется
    ``(1−k)²``, — а данных хватает, чтобы ``k`` держался около единицы.
    Поднять шум нельзя: цена уходит за ноль раньше, чем ошибка становится
    ощутимой.

    Тест сторожит именно это: если кто-то однажды решит оценивать раунд 4
    по прибыли, он сначала уронит вот эту проверку и прочитает, почему так
    делать не надо. Раунд оценивается рубрикой метода — заметил, проверил,
    дал честный интервал.
    """
    correct_q = (PARAMS.a - PARAMS.marginal_cost) / ((N_FIRMS + 1) * PARAMS.b)
    rivals = correct_q * (N_FIRMS - 1)

    def profit(own_q: float) -> float:
        price = PARAMS.a - PARAMS.b * (own_q + rivals)
        return (price - PARAMS.marginal_cost) * own_q

    best = profit(correct_q)
    shares = []
    for seed in SEEDS:
        exog, prices = _design(_case(seed))
        fit = sm.OLS(prices, exog).fit()
        intercept = float(fit.params[0])
        slope_hat = -float(fit.params[1])
        naive_q = (intercept - PARAMS.marginal_cost) / ((N_FIRMS + 1) * slope_hat)
        shares.append(profit(naive_q) / best)

    # Даже худший заход стоит меньше двух процентов прибыли.
    assert min(shares) > 0.98
