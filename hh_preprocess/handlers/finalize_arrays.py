"""Обработчик финализации данных, обработки пропусков и подготовки numpy-массивов."""

import logging

import numpy as np
import pandas as pd

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)

_TARGET_COL = "target_level"


class FinalizeArraysHandler(Handler):
    """Финализация датасета: проверки и сборка numpy-массивов."""

    _FILL_MEDIAN_COLS = ["age", "education_year"]

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Подготовить итоговые матрицы признаков (X) и таргета (y).

        Выполняет следующие действия:
        1. Заполняет пропуски в числовых колонках (возраст, год обучения) медианой.
        2. Удаляет строки, где целевая переменная не определена (NaN).
        3. Заполняет оставшиеся пропуски в признаках нулями.
        4. Преобразует DataFrame в numpy-массивы: X (float32) и y (int32).

        Аргументы:
            ctx: Контекст пайплайна с загруженным DataFrame.

        Возвращает:
            Обновлённый контекст пайплайна, в котором заполнены поля `ctx.X`,
            `ctx.y` и `ctx.feature_names`.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
            ValueError: Если в DataFrame нет целевой колонки.
            ValueError: Если не удалось преобразовать признаки в тип float32.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()

        if _TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{_TARGET_COL}' not found in DataFrame")

        ctx.diag["rows_before_finalize"] = int(df.shape[0])

        for col in self._FILL_MEDIAN_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                med = df[col].median(skipna=True)
                if pd.isna(med):
                    med = 0
                df[col] = df[col].fillna(med)

        before_drop = len(df)
        df = df.dropna(subset=[_TARGET_COL])
        dropped_count = before_drop - len(df)

        ctx.diag["dropped_rows_total"] = dropped_count
        ctx.diag["drop_reasons"] = {"nan_target": dropped_count}

        feature_cols = [c for c in df.columns if c != _TARGET_COL]

        nan_counts = df[feature_cols].isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            top_nans = (
                nan_counts[nan_counts > 0]
                .sort_values(ascending=False)
                .head(20)
                .to_dict()
            )
            ctx.diag["nan_report_top"] = top_nans
            logger.warning(
                f"Заполнение нулями оставшихся {total_nans} пропусков в признаках."
            )
            df[feature_cols] = df[feature_cols].fillna(0)

        try:
            X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        except Exception as e:
            bad_cols = {}
            for c in feature_cols:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    bad_cols[c] = str(df[c].dtype)
            ctx.diag["non_numeric_feature_cols"] = bad_cols
            raise ValueError(
                f"Не удалось преобразовать признаки в float32. Нечисловые колонки: {bad_cols}"
            ) from e

        y = df[_TARGET_COL].to_numpy(dtype=np.int32, copy=True)

        ctx.X = X
        ctx.y = y
        ctx.feature_names = feature_cols
        ctx.df = df

        logger.info("Final dataset: X=%s, y=%s", X.shape, y.shape)
        return ctx
