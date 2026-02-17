"""Обработчик загрузки исходных данных из CSV файла с поддержкой разных кодировок."""

import logging

import pandas as pd

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


class LoadCsvHandler(Handler):
    """Загрузить исходный CSV-файл в контекст пайплайна."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Прочитать CSV и записать DataFrame в контекст.

        Аргументы:
            ctx: Контекст пайплайна. Ожидается, что в нём задан `input_path`,
                а также опционально `sep` и `encoding`.

        Возвращает:
            Контекст с заполненными:
            - `ctx.df` — прочитанный DataFrame;
            - `ctx.diag` — метрики чтения.

        Исключения:
            RuntimeError: Если не удалось прочитать файл ни с одной из кодировок.
        """
        encodings = [ctx.encoding] if ctx.encoding else ["utf-8", "utf-8-sig", "cp1251"]
        last_err: Exception | None = None

        for enc in encodings:
            try:
                df = pd.read_csv(
                    ctx.input_path,
                    sep=ctx.sep,
                    engine="python",
                    encoding=enc,
                    on_bad_lines="skip",
                )
                ctx.df = df
                ctx.diag["input_rows"] = int(df.shape[0])
                ctx.diag["input_cols"] = int(df.shape[1])
                ctx.diag["csv_encoding_used"] = enc
                ctx.diag["csv_sep_used"] = ctx.sep if ctx.sep else "auto"
                logger.info(
                    "Loaded CSV: %s rows x %s cols (encoding=%s)",
                    df.shape[0],
                    df.shape[1],
                    enc,
                )
                return ctx
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"Failed to read CSV: {ctx.input_path}. Last error: {last_err}"
        )
