"""Обработчик извлечения уровня образования и года окончания учебного заведения."""

import logging
import re
from typing import Optional

import numpy as np

from ..context import PipelineContext
from ..utils.text import safe_lower
from .base import Handler

logger = logging.getLogger(__name__)

_EDU_COL = "Образование и ВУЗ"
_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
_UNKNOWN = "Не указано"

_LEVEL_MAP = {
    "Доктор наук": ["доктор"],
    "Кандидат наук": ["кандидат"],
    "Неоконченное высшее": ["неокончен", "incomplete higher"],
    "Высшее": ["высшее", "higher education", "bachelor", "master"],
    "Среднее специальное": ["среднее специаль", "college", "vocational"],
    "Среднее": ["среднее", "secondary"],
}


def _parse_level(val: object) -> str:
    """Определить укрупнённый уровень образования."""
    t = safe_lower(val)
    if not t:
        return _UNKNOWN

    for level, keywords in _LEVEL_MAP.items():
        if any(k in t for k in keywords):
            return level

    return _UNKNOWN


def _parse_year(val: object) -> float:
    """Извлечь год окончания."""
    t = safe_lower(val)
    if not t:
        return np.nan
    m = _YEAR_RE.search(t)
    if not m:
        return np.nan
    year = int(m.group(1))
    if 1950 <= year <= 2035:
        return float(year)
    return np.nan


class ParseEducationHandler(Handler):
    """Извлечь признаки образования в `education_level` и `education_year`."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Распарсить информацию об образовании и добавить признаки уровня и года.

        Из колонки `Образование и ВУЗ` извлекаются:
        - укрупнённый уровень образования (`education_level`) по ключевым словам;
        - год окончания (`education_year`)

        Если колонка отсутствует, устанавливаются значения по умолчанию:
        `education_level = "Не указано"`, `education_year = NaN`.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`, содержащим колонки
            `education_level` и `education_year`.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        if _EDU_COL not in df.columns:
            logger.warning("Column '%s' not found; using defaults.", _EDU_COL)
            df["education_level"] = _UNKNOWN
            df["education_year"] = np.nan
            ctx.df = df
            return ctx

        df["education_level"] = df[_EDU_COL].map(_parse_level)
        df["education_year"] = df[_EDU_COL].map(_parse_year)

        ctx.df = df
        return ctx
