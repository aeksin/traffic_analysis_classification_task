"""Обработчик извлечения общего опыта работы (преобразование в месяцы)."""

import logging
import re

from ..context import PipelineContext
from ..utils.text import safe_lower
from .base import Handler

logger = logging.getLogger(__name__)


_EXP_COL = "Опыт (двойное нажатие для полной версии)"


_YEARS_RE = re.compile(r"(\d+)\s*(?:год|года|лет|years|year)")
_MONTHS_RE = re.compile(r"(\d+)\s*(?:месяц|месяца|месяцев|months|month)")


def _parse_months(val: object) -> int:
    """Преобразовать текстовое описание опыта в количество месяцев.

    Аргументы:
        val: Значение из колонки с опытом работы.

    Возвращает:
        Общее количество месяцев опыта (целое число, не меньше нуля).
    """
    t = safe_lower(val)
    if not t:
        return 0
    if "нет опыта" in t or "no experience" in t:
        return 0

    years = 0
    months = 0

    my = _YEARS_RE.search(t)
    mm = _MONTHS_RE.search(t)

    if my:
        years = int(my.group(1))
    if mm:
        months = int(mm.group(1))

    total = years * 12 + months
    if total < 0:
        return 0
    return total


class ParseExperienceHandler(Handler):
    """Извлечь общий стаж работы и записать его в 'total_experience_months'."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Распарсить общий опыт работы и добавить признак в месяцах.

        Обработчик берёт колонку с опытом работы и извлекает из текста количество
        лет и месяцев (русские и английские варианты). Итоговое значение сохраняется
        в колонку 'total_experience_months'.

        Если ожидаемая колонка отсутствует, предпринимается попытка найти близкое
        совпадение среди колонок, начинающихся с «опыт» (без учёта регистра). Если
        подходящая колонка не найдена, используется значение '0' для всех строк.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым 'ctx.df', содержащим колонку
            'total_experience_months'.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        if _EXP_COL not in df.columns:
            # try to find a close match
            alt = next((c for c in df.columns if c.lower().startswith("опыт")), None)
            if alt is not None:
                exp_col = alt
            else:
                logger.warning("Experience column not found; using zeros.")
                df["total_experience_months"] = 0
                ctx.df = df
                return ctx
        else:
            exp_col = _EXP_COL

        df["total_experience_months"] = df[exp_col].map(_parse_months).astype(int)

        ctx.df = df
        return ctx
