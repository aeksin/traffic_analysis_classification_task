"""Обработчик One-Hot кодирования категориальных признаков."""

import logging

import numpy as np
import pandas as pd

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)

_RAW_TEXT_COL_HINTS = [
    "пол, возраст",
    "зп",
    "город",
    "занятость",
    "график",
    "опыт",
    "образование",
    "обновление резюме",
    "авто",
]


class OneHotEncodeHandler(Handler):
    """Закодировать категориальные признаки методом one-hot."""

    _NUMERIC_COLS = (
        "age",
        "education_year",
        "resume_days_since_update",
        "target_level",
        "description_len_log",
        "age_sq",
    )

    _LEAK_COLS = ["total_experience_months", "experience_sq", "target_salary_rub"]

    def _drop_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удалить исходные текстовые колонки и признаки-утечки."""
        drop_cols = []
        for c in df.columns:
            cl = str(c).lower()
            if any(h in cl for h in _RAW_TEXT_COL_HINTS):
                if c not in ["target_salary_rub", "target_level"]:
                    drop_cols.append(c)

        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
            logger.info("Dropped raw columns: %d", len(drop_cols))

        # 2. Удаляем специфичные для задачи утечки (leakage)
        leaks_to_drop = [c for c in self._LEAK_COLS if c in df.columns]
        if leaks_to_drop:
            logger.info("Dropping target leaks: %s", leaks_to_drop)
            df = df.drop(columns=leaks_to_drop, errors="ignore")

        return df

    def _convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Привести числовые колонки к типу numeric и bool к int."""
        for col in self._NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Bool -> int
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.int8)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применить get_dummies к категориальным колонкам."""
        cat_cols = []
        for c in df.columns:
            # Пропускаем таргеты и числа
            if c in ["target_salary_rub", "target_level"]:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                continue

            # Отбираем object/string/category
            if (
                df[c].dtype == object
                or pd.api.types.is_string_dtype(df[c])
                or pd.api.types.is_categorical_dtype(df[c])
            ):
                cat_cols.append(c)

        if cat_cols:
            df[cat_cols] = df[cat_cols].fillna("Не указано")
            df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
        return df

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Подготовить DataFrame к сохранению в numpy-массивы."""
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()

        # Делегируем логику вспомогательным методам для снижения сложности
        df = self._drop_raw_columns(df)
        df = self._convert_numeric(df)
        df = self._encode_categoricals(df)

        ctx.df = df
        return ctx
