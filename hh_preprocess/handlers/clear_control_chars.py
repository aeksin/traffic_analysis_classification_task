"""Обработчик очистки текста от управляющих символов (BOM, NBSP и др.)."""

import re

import pandas as pd
from pandas.api.types import (is_categorical_dtype, is_object_dtype,
                              is_string_dtype)

from ..context import PipelineContext
from .base import Handler

_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")  # управляющие ASCII (0..31) + DEL (127)


class CleanControlCharsHandler(Handler):
    """Очистить управляющие символы и нормализовать пробелы в текстовых столбцах.

    Убираем:
    - BOM (\\ufeff)
    - NBSP (\\xa0)
    - управляющие символы
    """

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Очистить строки во всех текстовых столбцах.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Обновлённый контекст пайплайна.
        """
        if ctx.df is None:
            raise ValueError(
                "В контексте отсутствует df. Проверьте порядок обработчиков."
            )

        df = ctx.df

        text_cols: list[str] = []
        for col in df.columns:
            s = df[col]
            if (
                is_object_dtype(s.dtype)
                or is_string_dtype(s.dtype)
                or is_categorical_dtype(s.dtype)
            ):
                text_cols.append(col)

        replaced_cells = 0

        def _clean_cell(x: object) -> object:
            nonlocal replaced_cells
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return x
            if not isinstance(x, str):
                return x

            orig = x
            # BOM / NBSP
            x = x.replace("\ufeff", "").replace("\xa0", " ")
            # управляющие символы -> пробел
            x = _CONTROL_RE.sub(" ", x)
            # множественные пробелы
            x = " ".join(x.split())

            if x != orig:
                replaced_cells += 1
            return x

        if text_cols:
            df[text_cols] = df[text_cols].map(_clean_cell)

        ctx.df = df
        ctx.diag.setdefault("clean_control_chars", {})
        ctx.diag["clean_control_chars"] = {
            "text_cols": len(text_cols),
            "cells_changed": replaced_cells,
        }
        return ctx
