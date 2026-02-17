"""Обработчик парсинга зарплаты, определения валюты и конвертации в рубли."""

import logging
import re
from typing import Optional

import numpy as np

from ..context import PipelineContext
from ..utils.currency import load_fx_rates
from ..utils.text import normalize_spaces, safe_lower
from .base import Handler

logger = logging.getLogger(__name__)

_SALARY_COL = "ЗП"
_NUM_RE = re.compile(r"(\d[\d\s]*)")

_CURRENCY_MAP = {
    "руб": "RUB",
    "rur": "RUB",
    "rub": "RUB",
    "₽": "RUB",
    "usd": "USD",
    "$": "USD",
    "eur": "EUR",
    "€": "EUR",
    "kzt": "KZT",
    "тенге": "KZT",
    "byn": "BYN",
    "бел": "BYN",
    "uah": "UAH",
    "грн": "UAH",
    "uzs": "UZS",
    "сум": "UZS",
    "gel": "GEL",
    "лари": "GEL",
    "amd": "AMD",
    "драм": "AMD",
    "azn": "AZN",
    "манат": "AZN",
}


def _detect_currency(s: str) -> Optional[str]:
    """Определить валюту по текстовому представлению зарплаты.

    По строке ищутся характерные маркеры валюты (например, `руб`, `usd`, `€`).
    Если валюта не распознана, возвращается `None`.

    Аргументы:
        s: Текстовое значение зарплаты.

    Возвращает:
        Код валюты (например, `RUB`, `USD`, `EUR`) или `None`, если определить не удалось.
    """
    t = s.lower()
    for key, code in _CURRENCY_MAP.items():
        if key in t:
            return code
    return None


def _extract_numbers(s: str) -> list[int]:
    """Извлечь целые числа из строки.

    По регулярному выражению находятся группы цифр (в том числе с пробелами и
    неразрывными пробелами), после чего они нормализуются и преобразуются в `int`.

    Аргументы:
        s: Текст, в котором нужно найти числа.

    Возвращает:
        Список найденных чисел в порядке появления в строке.
    """
    nums: list[int] = []
    for m in _NUM_RE.finditer(s):
        raw = m.group(1).replace("\u00a0", " ").replace(" ", "")
        if raw.isdigit():
            nums.append(int(raw))
    return nums


def _calculate_rub_salary(val: object, rates: dict) -> tuple[float, str]:
    """Вычислить зарплату в рублях для одной строки."""
    if not isinstance(val, str):
        return np.nan, ""

    s = normalize_spaces(val)
    if not s:
        return np.nan, ""

    t = safe_lower(s)
    if any(stop in t for stop in ["договор", "negotiable"]):
        return np.nan, ""

    cur = _detect_currency(s) or "RUB"
    nums = _extract_numbers(s)

    if not nums:
        return np.nan, cur

    amount = float(nums[0])
    if len(nums) >= 2:
        amount = (nums[0] + nums[1]) / 2.0

    if "тыс" in t or "k" in t:
        amount *= 1000.0

    rate = rates.get(cur)
    if rate is None:
        return np.nan, cur

    rub = amount * float(rate)
    return (rub, cur) if rub > 0 else (np.nan, cur)


class ParseSalaryHandler(Handler):
    """Преобразовать зарплату в рублях в колонку `target_salary_rub`."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Распарсить зарплату и вычислить целевую зарплату в рублях.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`, содержащим колонки
            `salary_currency` и `target_salary_rub`.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
            ValueError: Если в DataFrame отсутствует обязательная колонка `ЗП`.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()
        if _SALARY_COL not in df.columns:
            raise ValueError(f"Required target column '{_SALARY_COL}' not found")

        fx = load_fx_rates(ctx.output_dir)
        ctx.diag["fx_rates_source"] = fx.source

        results = df[_SALARY_COL].apply(lambda x: _calculate_rub_salary(x, fx.rates))

        df["target_salary_rub"] = [r[0] for r in results]
        df["salary_currency"] = [r[1] for r in results]

        target = df["target_salary_rub"]
        nan_mask = target.isna()
        ctx.diag["rows_with_nan_target"] = int(nan_mask.sum())
        if nan_mask.any():
            ex = df.loc[nan_mask, [_SALARY_COL, "salary_currency"]].head(20).copy()
            ctx.diag["nan_target_examples"] = ex

        ctx.diag["target_zeros"] = int((target == 0).sum())

        ctx.df = df
        return ctx
