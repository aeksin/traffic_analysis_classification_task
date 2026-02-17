"""Обработчик финализации данных, обработки пропусков и подготовки numpy-массивов."""

import logging

import numpy as np
import pandas as pd

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


_TARGET_COL = "target_salary_rub"


class FinalizeArraysHandler(Handler):
    """Финализация датасета: проверки и сборка numpy-массивов."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()
        if _TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{_TARGET_COL}' not found")

        ctx.diag["rows_before_finalize"] = int(df.shape[0])

        for col in ["age", "education_year"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                med = df[col].median(skipna=True)
                df[col] = df[col].fillna(med)

        nan_target = df[_TARGET_COL].isna()
        dropped_nan_target = int(nan_target.sum())
        if dropped_nan_target:
            df = df.loc[~nan_target].copy()

        zeros_target = int((df[_TARGET_COL] == 0).sum())
        if zeros_target:
            df = df.loc[df[_TARGET_COL] != 0].copy()

        ctx.diag["dropped_rows_total"] = int(
            ctx.diag.get("rows_before_finalize", 0) - df.shape[0]
        )
        ctx.diag["drop_reasons"] = {
            "nan_target": dropped_nan_target,
            "zero_target": zeros_target,
        }

        nan_counts = df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        if len(nan_counts):
            top = nan_counts.sort_values(ascending=False).head(20).to_dict()
            ctx.diag["nan_report_top"] = top
            raise ValueError(f"Неожиданные NaN после обработки (top 20): {top}")

        feature_cols = [c for c in df.columns if c != _TARGET_COL]

        try:
            X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        except Exception as e:
            bad_cols: dict[str, str] = {}
            for c in feature_cols:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    bad_cols[c] = str(df[c].dtype)
            ctx.diag["non_numeric_feature_cols"] = bad_cols
            raise ValueError(
                "Не удалось преобразовать признаки в float32. "
                "Проверьте one-hot кодирование и типы колонок. "
                f"Нечисловые колонки: {bad_cols}"
            ) from e

        y = df[_TARGET_COL].to_numpy(dtype=np.float32, copy=True)

        ctx.X = X
        ctx.y = y
        ctx.feature_names = feature_cols
        ctx.df = df

        logger.info("Final dataset: X=%s, y=%s", X.shape, y.shape)
        return ctx
