"""Обработчик парсинга типа занятости и графика работы."""

import logging

from ..context import PipelineContext
from ..utils.text import split_multi_categories
from .base import Handler

logger = logging.getLogger(__name__)


_EMP_COL = "Занятость"
_SCH_COL = "График"


_EMP_CANON = {
    "полная занятость": "emp_full",
    "частичная занятость": "emp_part",
    "проектная работа": "emp_project",
    "стажировка": "emp_intern",
    "волонтерство": "emp_volunteer",
}

_SCH_CANON = {
    "полный день": "sch_full_day",
    "сменный график": "sch_shift",
    "гибкий график": "sch_flexible",
    "удаленная работа": "sch_remote",
    "вахтовый метод": "sch_rotation",
}


def _canon_emp(token: str) -> str | None:
    """Нормализовать значение занятости.

    Аргументы:
        token: Один элемент (категория) из исходного текста.

    Возвращает:
        Каноническое русскоязычное название категории (ключ из `_EMP_CANON`)
        или `None`, если сопоставление не найдено.
    """
    t = token.lower()
    if "full time" in t:
        return "полная занятость"
    if "part time" in t:
        return "частичная занятость"
    if "project" in t:
        return "проектная работа"
    if "intern" in t or "стажиров" in t:
        return "стажировка"
    if "volunteer" in t or "волонтер" in t:
        return "волонтерство"

    for k in _EMP_CANON:
        if k in t:
            return k
    return None


def _canon_sch(token: str) -> str | None:
    """Нормализовать значение графика работы.

    Пытается сопоставить произвольный текстовый токен с одной из поддерживаемых
    категорий графика (русские/английские варианты и подстроки).

    Аргументы:
        token: Один элемент (категория) из исходного текста.

    Возвращает:
        Каноническое русскоязычное название категории (ключ из `_SCH_CANON`)
        или `None`, если сопоставление не найдено.
    """
    t = token.lower()
    if "remote" in t or "удален" in t:
        return "удаленная работа"
    if "flex" in t or "гибк" in t:
        return "гибкий график"
    if "shift" in t or "сменн" in t:
        return "сменный график"
    if "rotation" in t or "вахт" in t:
        return "вахтовый метод"
    if "full day" in t or "полный день" in t:
        return "полный день"
    for k in _SCH_CANON:
        if k in t:
            return k
    return None


class ParseEmploymentScheduleHandler(Handler):
    """Распарсить занятость и график и сформировать признаки `employment_*` и `schedule_*`."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Сформировать one-hot признаки занятости и графика работы.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`, содержащим one-hot колонки
            для занятости и графика.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        for col in _EMP_CANON.values():
            df[col] = 0
        for col in _SCH_CANON.values():
            df[col] = 0

        if _EMP_COL in df.columns:
            for idx, val in df[_EMP_COL].items():
                for token in split_multi_categories(val):
                    canon = _canon_emp(token)
                    if canon is None:
                        continue
                    df.at[idx, _EMP_CANON[canon]] = 1

        if _SCH_COL in df.columns:
            for idx, val in df[_SCH_COL].items():
                for token in split_multi_categories(val):
                    canon = _canon_sch(token)
                    if canon is None:
                        continue
                    df.at[idx, _SCH_CANON[canon]] = 1

        ctx.df = df
        return ctx
