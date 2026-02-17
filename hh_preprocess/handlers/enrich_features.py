"""Обработчик генерации обогащенных признаков."""

import logging

import numpy as np
import pandas as pd

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


class EnrichFeaturesHandler(Handler):
    """Генерация новых признаков: полиномы, взаимодействия и текстовые метрики."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Создать новые признаки на основе существующих.

        Добавляет:
        - Квадраты возраста и опыта (для учета нелинейности).
        - Логарифм длины описания опыта.
        - Комбинацию города и категории вакансии (interaction feature).

        Аргументы:
            ctx: Контекст пайплайна с загруженным DataFrame.

        Возвращает:
            Обновленный контекст.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()
        logger.info("Генерация обогащенных признаков (EnrichFeatures)...")

        if "age" in df.columns:
            age_filled = df["age"].fillna(df["age"].median())
            df["age_sq"] = age_filled**2

        if "total_experience_months" in df.columns:
            df["experience_sq"] = df["total_experience_months"] ** 2

        exp_text_col = next(
            (c for c in df.columns if str(c).lower().startswith("опыт")), None
        )
        if exp_text_col:
            text_len = df[exp_text_col].astype(str).str.len()
            df["description_len_log"] = np.log1p(text_len)

        if "city_group" in df.columns and "job_category" in df.columns:
            df["geo_job"] = (
                df["city_group"].astype(str) + "_" + df["job_category"].astype(str)
            )

        ctx.df = df
        return ctx
