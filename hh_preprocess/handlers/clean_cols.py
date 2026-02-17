"""Обработчик очистки названий колонок от лишних пробелов и артефактов."""

import logging

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


class CleanColumnsHandler(Handler):
    """Очистить названия колонок DataFrame.

    Удаляет пробелы по краям названий колонок и удаляет безымянные колонки
    (например, 'Unnamed: 0'), возникающие при сохранении index в CSV.
    """

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Выполнить очистку названий колонок.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Обновленный контекст с очищенными именами колонок в df.

        Исключения:
            ValueError: Если DataFrame не загружен в контекст.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()

        df.columns = [str(c).strip() for c in df.columns]

        drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
            logger.info("Dropped columns: %s", drop_cols)

        ctx.df = df
        return ctx
