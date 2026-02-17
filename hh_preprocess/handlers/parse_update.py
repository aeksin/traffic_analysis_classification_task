"""Обработчик вычисления давности обновления резюме в днях."""

import logging

import pandas as pd

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


_UPDATE_COL = "Обновление резюме"


class ParseResumeUpdateHandler(Handler):
    """Преобразовать дату обновления резюме в `resume_days_since_update`."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Распарсить дату обновления резюме и посчитать дни с момента обновления.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`, содержащим колонку
            `resume_days_since_update`.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        if _UPDATE_COL not in df.columns:
            logger.warning("Column '%s' not found; using zeros.", _UPDATE_COL)
            df["resume_days_since_update"] = 0
            ctx.df = df
            return ctx

        dt = pd.to_datetime(df[_UPDATE_COL], errors="coerce", dayfirst=True)
        ref = dt.max()
        if pd.isna(ref):
            ref = pd.Timestamp.utcnow()

        if dt.isna().any():
            mode = dt.mode()
            fill_val = mode.iloc[0] if len(mode) else ref
            dt = dt.fillna(fill_val)

        days = (ref - dt).dt.days.astype(int)
        days = days.clip(lower=0)

        df["resume_days_since_update"] = days

        ctx.df = df
        return ctx
