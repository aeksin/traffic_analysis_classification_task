"""Обработчик парсинга типа занятости и графика работы."""

import logging
from typing import Any, Dict

import pandas as pd

from ..context import PipelineContext
from ..utils.text import split_multi_categories
from .base import Handler

logger = logging.getLogger(__name__)


_EMP_COL = "Занятость"
_SCH_COL = "График"

_FULL_DAY_VAL = "полный день"

_EMP_CANON = {
    "полная занятость": "emp_full",
    "частичная занятость": "emp_part",
    "проектная работа": "emp_project",
    "стажировка": "emp_intern",
    "волонтерство": "emp_volunteer",
}

_SCH_CANON = {
    _FULL_DAY_VAL: "sch_full_day",
    "сменный график": "sch_shift",
    "гибкий график": "sch_flexible",
    "удаленная работа": "sch_remote",
    "вахтовый метод": "sch_rotation",
}


def _canon_emp(token: str) -> str | None:
    """Нормализовать значение занятости."""
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
    """Нормализовать значение графика работы."""
    t = token.lower()
    if "remote" in t or "удален" in t:
        return "удаленная работа"
    if "flex" in t or "гибк" in t:
        return "гибкий график"
    if "shift" in t or "сменн" in t:
        return "сменный график"
    if "rotation" in t or "вахт" in t:
        return "вахтовый метод"
    if "full day" in t or _FULL_DAY_VAL in t:
        return _FULL_DAY_VAL
    for k in _SCH_CANON:
        if k in t:
            return k
    return None


class ParseEmploymentScheduleHandler(Handler):
    """Распарсить занятость и график и сформировать признаки `employment_*` и `schedule_*`."""

    def _process_column(
        self,
        df: pd.DataFrame,
        col_name: str,
        canon_map: Dict[str, str],
        canon_func: Any,
    ) -> None:
        """Обработать одну колонку (Занятость или График) и проставить 1 в соответствующие поля."""
        if col_name not in df.columns:
            return

        # Итерируемся по строкам, где есть значения
        series = df[col_name].dropna()
        for idx, val in series.items():
            # Разбиваем "полная занятость, удаленная работа" на части
            tokens = split_multi_categories(val)
            for token in tokens:
                canon = canon_func(token)
                if canon is not None and canon in canon_map:
                    target_col = canon_map[canon]
                    df.at[idx, target_col] = 1

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Сформировать one-hot признаки занятости и графика работы.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`.

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

        # Обработка Занятости
        self._process_column(df, _EMP_COL, _EMP_CANON, _canon_emp)

        # Обработка Графика
        self._process_column(df, _SCH_COL, _SCH_CANON, _canon_sch)

        ctx.df = df
        return ctx
