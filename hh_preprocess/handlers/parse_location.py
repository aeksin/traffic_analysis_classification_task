"""Обработчик извлечения города проживания и признаков готовности к релокации."""

import logging

from ..context import PipelineContext
from ..utils.text import normalize_spaces, safe_lower
from .base import Handler

logger = logging.getLogger(__name__)

_CITY_COL = "Город"
_UNKNOWN = "Не указано"


def _parse_city(s: object) -> str:
    """Извлечь город проживания."""
    if not isinstance(s, str):
        return _UNKNOWN
    t = normalize_spaces(s)
    if not t:
        return _UNKNOWN
    return normalize_spaces(t.split(",")[0])


def _check_keywords(text: str, yes_keys: list[str], no_keys: list[str]) -> bool:
    """Универсальная проверка на вхождение ключевых фраз."""
    if not text:
        return False
    if any(k in text for k in no_keys):
        return False
    if any(k in text for k in yes_keys):
        return True
    return False


class ParseLocationHandler(Handler):
    """Извлечь признаки города, готовности к переезду и командировкам."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Распарсить колонку «Город» и сформировать признаки мобильности.

        Из исходной колонки формируются признаки:
        - `city` — город проживания (первая часть строки до запятой);
        - `relocate_ready` — готовность к переезду (bool);
        - `trips_ready` — готовность к командировкам (bool).

        Если исходная колонка отсутствует, создаются значения по умолчанию:
        - `city = "Не указано"`;
        - `relocate_ready = False`;
        - `trips_ready = False`.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`, содержащим колонки
            `city`, `relocate_ready`, `trips_ready`.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        if _CITY_COL not in df.columns:
            logger.warning("Column '%s' not found; creating defaults.", _CITY_COL)
            df["city"] = _UNKNOWN
            df["relocate_ready"] = False
            df["trips_ready"] = False
            ctx.df = df
            return ctx

        norm_col = df[_CITY_COL].map(safe_lower)

        df["city"] = df[_CITY_COL].map(_parse_city)

        df["relocate_ready"] = norm_col.apply(
            lambda t: _check_keywords(
                t,
                yes_keys=[
                    "готов к переезду",
                    "ready to relocate",
                    "willing to relocate",
                ],
                no_keys=["не готов к переезду", "not ready", "not willing"],
            )
        ).astype(bool)

        df["trips_ready"] = norm_col.apply(
            lambda t: _check_keywords(
                t,
                yes_keys=[
                    "готов к командировкам",
                    "ready for business trips",
                    "willing to travel",
                ],
                no_keys=["не готов к командировкам", "not ready for business trips"],
            )
        ).astype(bool)

        ctx.df = df
        return ctx
