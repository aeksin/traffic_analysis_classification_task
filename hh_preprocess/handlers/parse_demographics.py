"""Обработчик извлечения пола и возраста из сырых данных."""

import logging
import re

import numpy as np

from ..context import PipelineContext
from ..utils.text import safe_lower
from .base import Handler

logger = logging.getLogger(__name__)


_GENDER_AGE_COL = "Пол, возраст"


def _parse_gender(s: object) -> str:
    """Определить пол по текстовому значению.

    Аргументы:
        s: Значение из колонки «Пол, возраст».

    Возвращает:
        Строковая метка пола: `"M"`, `"F"` или `"U"`.
    """
    t = safe_lower(s)
    if "муж" in t or "male" in t:
        return "M"
    if "жен" in t or "female" in t:
        return "F"
    return "U"


def _parse_age(s: object) -> float:
    """Извлечь возраст из текстового значения.

    Аргументы:
        s: Значение из колонки «Пол, возраст».

    Возвращает:
        Возраст в годах (float) или `NaN`, если возраст не распознан
        либо выходит за допустимые границы.
    """
    t = safe_lower(s)
    m = re.search(r"(\d{1,3})\s*(?:лет|года|год|years|year)", t)
    if not m:
        m2 = re.search(r"(\d{1,3})", t)
        if not m2:
            return np.nan
        val = int(m2.group(1))
    else:
        val = int(m.group(1))

    if 0 <= val <= 100:
        return float(val)
    return np.nan


class ParseGenderAgeHandler(Handler):
    """Извлечь пол и возраст в признаки 'gender' и 'age'."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Распарсить колонку «Пол, возраст» и добавить признаки пола и возраста.

        Аргументы:
            ctx: Контекст пайплайна. Ожидается, что `ctx.df` уже загружен.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`, содержащим колонки `gender` и `age`.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        if _GENDER_AGE_COL not in df.columns:
            logger.warning(
                "Column '%s' not found; creating default gender/age.", _GENDER_AGE_COL
            )
            df["gender"] = "U"
            df["age"] = np.nan
            ctx.df = df
            return ctx

        raw = df[_GENDER_AGE_COL].astype(object)

        df["gender"] = raw.map(_parse_gender)
        df["age"] = raw.map(_parse_age)

        ctx.df = df
        return ctx
